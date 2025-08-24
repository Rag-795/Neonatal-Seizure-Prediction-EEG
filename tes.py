import mne
import pandas as pd
import numpy as np
import scipy.io
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class MNENeonatalSeizureExtractor:
    def __init__(self, dataset_path="dataset/"):
        self.dataset_path = Path(dataset_path)
        self.eeg_files = []
        self.annotation_file = None
        self.seizure_annotations = {}
        
    def find_files(self):
        """Find EEG files and annotation file in dataset"""
        print("=" * 60)
        print("FINDING EEG AND ANNOTATION FILES")
        print("=" * 60)
        
        # Find EEG files (.edf)
        eeg_pattern = str(self.dataset_path / "*.edf")
        self.eeg_files = glob.glob(eeg_pattern)
        self.eeg_files.sort()
        
        print(f"Found {len(self.eeg_files)} EEG files:")
        for i, file in enumerate(self.eeg_files[:5]):  # Show first 5
            print(f"  {i+1}. {Path(file).name}")
        if len(self.eeg_files) > 5:
            print(f"  ... and {len(self.eeg_files) - 5} more files")
        
        # Find annotation file (.mat)
        mat_files = glob.glob(str(self.dataset_path / "*.mat"))
        if mat_files:
            self.annotation_file = mat_files[0]
            print(f"\nFound annotation file: {Path(self.annotation_file).name}")
        else:
            print("\n⚠ No .mat annotation file found!")
            
        return len(self.eeg_files) > 0 and self.annotation_file is not None
    
    def load_matlab_annotations(self):
        """Load and parse MATLAB annotation file"""
        print("\n" + "=" * 60)
        print("LOADING MATLAB ANNOTATIONS")
        print("=" * 60)
        
        try:
            # Load MATLAB file
            mat_data = scipy.io.loadmat(self.annotation_file)
            
            # Display structure of MATLAB file
            print("MATLAB file structure:")
            for key, value in mat_data.items():
                if not key.startswith('__'):
                    print(f"  {key}: {type(value)} {getattr(value, 'shape', '')}")
            
            # Common keys in neonatal seizure datasets
            possible_keys = ['annotations', 'seizures', 'events', 'labels', 'data']
            annotation_data = None
            
            for key in possible_keys:
                if key in mat_data:
                    annotation_data = mat_data[key]
                    print(f"\nUsing '{key}' as annotation data")
                    break
            
            if annotation_data is None:
                # Use the largest non-system variable
                non_system_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if non_system_keys:
                    key = max(non_system_keys, key=lambda k: np.prod(mat_data[k].shape) if hasattr(mat_data[k], 'shape') else 0)
                    annotation_data = mat_data[key]
                    print(f"\nUsing '{key}' as annotation data (largest variable)")
            
            return self.parse_annotation_data(annotation_data)
            
        except Exception as e:
            print(f"✗ Error loading MATLAB file: {e}")
            return False
    
    def parse_annotation_data(self, annotation_data):
        """Parse annotation data structure"""
        try:
            # Handle different MATLAB data structures
            if isinstance(annotation_data, np.ndarray):
                if annotation_data.dtype.names:  # Structured array
                    return self.parse_structured_array(annotation_data)
                else:  # Regular array
                    return self.parse_regular_array(annotation_data)
            else:
                print(f"Unexpected annotation data type: {type(annotation_data)}")
                return False
                
        except Exception as e:
            print(f"✗ Error parsing annotation data: {e}")
            return False
    
    def parse_structured_array(self, data):
        """Parse structured MATLAB array"""
        print(f"Structured array with fields: {data.dtype.names}")
        
        # Common field names for seizure annotations
        time_fields = ['onset', 'start', 'time', 'onset_time', 'start_time']
        duration_fields = ['duration', 'length', 'end', 'stop']
        patient_fields = ['patient', 'subject', 'record', 'eeg', 'file']
        expert_fields = ['expert', 'annotator', 'rater', 'observer']
        
        seizure_data = []
        
        for record in data.flatten():
            seizure_info = {}
            
            # Extract available fields
            for field_name in record.dtype.names:
                value = record[field_name]
                
                # Handle nested structures or cell arrays
                if hasattr(value, 'flatten'):
                    if value.size > 0:
                        flat_val = value.flatten()[0] if value.size == 1 else value.flatten()
                        seizure_info[field_name] = flat_val
                    else:
                        seizure_info[field_name] = None
                else:
                    seizure_info[field_name] = value
            
            if seizure_info:  # Only add non-empty records
                seizure_data.append(seizure_info)
        
        # Convert to DataFrame
        if seizure_data:
            df = pd.DataFrame(seizure_data)
            print(f"✓ Parsed {len(df)} annotation records")
            print(f"Columns: {list(df.columns)}")
            return df
        else:
            print("⚠ No annotation records found")
            return pd.DataFrame()
    
    def parse_regular_array(self, data):
        """Parse regular MATLAB array"""
        print(f"Regular array shape: {data.shape}")
        
        # Assume columns are: [patient_id, onset_time, duration, ...]
        if data.ndim == 2 and data.shape[1] >= 3:
            columns = ['patient_id', 'onset_time', 'duration']
            if data.shape[1] > 3:
                columns.extend([f'col_{i}' for i in range(3, data.shape[1])])
            
            df = pd.DataFrame(data, columns=columns)
            print(f"✓ Created DataFrame with {len(df)} records")
            return df
        else:
            print(f"⚠ Unexpected array dimensions: {data.shape}")
            return pd.DataFrame()
    
    def load_eeg_file(self, eeg_path):
        """Load EEG file using MNE"""
        try:
            # Load EDF file
            raw = mne.io.read_raw_edf(eeg_path, preload=True, verbose=False)
            
            # Get basic info
            info = {
                'filename': Path(eeg_path).name,
                'duration': raw.times[-1],
                'sampling_rate': raw.info['sfreq'],
                'n_channels': len(raw.ch_names),
                'channels': raw.ch_names[:5]  # First 5 channels
            }
            
            return raw, info
            
        except Exception as e:
            print(f"✗ Error loading {Path(eeg_path).name}: {e}")
            return None, None
    
    def extract_seizure_patients(self, annotations_df):
        """Extract patients with seizures from annotations"""
        print("\n" + "=" * 60)
        print("EXTRACTING SEIZURE PATIENTS")
        print("=" * 60)
        
        if annotations_df.empty:
            print("⚠ No annotations available")
            return []
        
        # Find patient identifier column
        patient_cols = []
        for col in annotations_df.columns:
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in ['patient', 'subject', 'record', 'eeg', 'file']):
                patient_cols.append(col)
        
        if not patient_cols:
            # Use first column as patient identifier
            patient_cols = [annotations_df.columns[0]]
            print(f"Using '{patient_cols[0]}' as patient identifier")
        
        seizure_patients = []
        for col in patient_cols:
            patients = annotations_df[col].dropna().unique()
            seizure_patients.extend([str(p) for p in patients])
        
        seizure_patients = list(set(seizure_patients))  # Remove duplicates
        
        print(f"✓ Found {len(seizure_patients)} unique patients with seizures")
        return seizure_patients
    
    def match_eeg_files_to_patients(self, seizure_patients):
        """Match EEG files to seizure patients"""
        print("\n" + "=" * 60)
        print("MATCHING EEG FILES TO SEIZURE PATIENTS")
        print("=" * 60)
        
        matched_files = {}
        
        for eeg_file in self.eeg_files:
            filename = Path(eeg_file).stem  # Filename without extension
            
            # Extract patient number from filename (e.g., eeg1.edf -> 1)
            patient_num = ''.join(filter(str.isdigit, filename))
            
            if patient_num in seizure_patients:
                matched_files[patient_num] = eeg_file
                print(f"✓ Patient {patient_num}: {Path(eeg_file).name}")
            elif filename in seizure_patients:
                matched_files[filename] = eeg_file
                print(f"✓ Patient {filename}: {Path(eeg_file).name}")
        
        print(f"\n✓ Matched {len(matched_files)} EEG files to seizure patients")
        return matched_files
    
    def process_seizure_data(self, annotations_df, matched_files):
        """Process seizure data for each patient"""
        print("\n" + "=" * 60)
        print("PROCESSING SEIZURE DATA")
        print("=" * 60)
        
        seizure_data_list = []
        
        for patient_id, eeg_file in matched_files.items():
            print(f"\nProcessing Patient {patient_id}...")
            
            # Load EEG data
            raw, eeg_info = self.load_eeg_file(eeg_file)
            if raw is None:
                continue
            
            # Find annotations for this patient
            patient_annotations = self.get_patient_annotations(annotations_df, patient_id)
            
            if not patient_annotations.empty:
                print(f"  ✓ Found {len(patient_annotations)} seizure annotations")
                
                # Add EEG info to annotations
                for idx, annotation in patient_annotations.iterrows():
                    seizure_record = annotation.to_dict()
                    seizure_record.update({
                        'eeg_filename': eeg_info['filename'],
                        'eeg_duration': eeg_info['duration'],
                        'sampling_rate': eeg_info['sampling_rate'],
                        'n_channels': eeg_info['n_channels'],
                        'patient_id': patient_id
                    })
                    seizure_data_list.append(seizure_record)
            else:
                print(f"  ⚠ No annotations found for Patient {patient_id}")
        
        if seizure_data_list:
            seizure_df = pd.DataFrame(seizure_data_list)
            print(f"\n✓ Processed seizure data for {len(matched_files)} patients")
            print(f"✓ Total seizure records: {len(seizure_df)}")
            return seizure_df
        else:
            return pd.DataFrame()
    
    def get_patient_annotations(self, annotations_df, patient_id):
        """Get annotations for specific patient"""
        # Try different ways to match patient ID
        for col in annotations_df.columns:
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in ['patient', 'subject', 'record', 'eeg', 'file']):
                # Try exact match
                patient_data = annotations_df[annotations_df[col] == patient_id]
                if not patient_data.empty:
                    return patient_data
                
                # Try string match
                patient_data = annotations_df[annotations_df[col].astype(str) == patient_id]
                if not patient_data.empty:
                    return patient_data
                
                # Try partial match (e.g., "eeg1" contains "1")
                patient_data = annotations_df[annotations_df[col].astype(str).str.contains(patient_id, na=False)]
                if not patient_data.empty:
                    return patient_data
        
        return pd.DataFrame()
    
    def save_seizure_csv(self, seizure_df, output_path="seizure_patients_annotations.csv"):
        """Save seizure data to CSV"""
        try:
            seizure_df.to_csv(output_path, index=False)
            print(f"✓ Seizure data saved to: {output_path}")
            
            # Create summary
            self.create_summary_report(seizure_df, output_path)
            return True
            
        except Exception as e:
            print(f"✗ Error saving CSV: {e}")
            return False
    
    def create_summary_report(self, seizure_df, csv_path):
        """Create summary report"""
        report_path = csv_path.replace('.csv', '_summary.txt')
        
        with open(report_path, 'w') as f:
            f.write("MNE NEONATAL EEG SEIZURE DATA SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total seizure annotations: {len(seizure_df)}\n")
            f.write(f"Unique patients with seizures: {seizure_df['patient_id'].nunique()}\n\n")
            
            f.write("Patient seizure counts:\n")
            patient_counts = seizure_df['patient_id'].value_counts().sort_index()
            for patient, count in patient_counts.items():
                f.write(f"  Patient {patient}: {count} seizures\n")
            
            f.write(f"\nColumns in dataset:\n")
            for col in seizure_df.columns:
                f.write(f"  - {col}\n")
            
            if 'eeg_duration' in seizure_df.columns:
                f.write(f"\nEEG recording statistics:\n")
                f.write(f"  Average duration: {seizure_df['eeg_duration'].mean():.1f} seconds\n")
                f.write(f"  Total recording time: {seizure_df['eeg_duration'].sum():.1f} seconds\n")
        
        print(f"✓ Summary report saved to: {report_path}")
    
    def run_extraction(self):
        """Main extraction process"""
        print("MNE-BASED NEONATAL EEG SEIZURE DATA EXTRACTION")
        print("=" * 60)
        
        # Step 1: Find files
        if not self.find_files():
            print("✗ Required files not found!")
            return None
        
        # Step 2: Load annotations
        annotations_df = self.load_matlab_annotations()
        if annotations_df is None or annotations_df.empty:
            print("✗ Failed to load annotations!")
            return None
        
        # Step 3: Extract seizure patients
        seizure_patients = self.extract_seizure_patients(annotations_df)
        if not seizure_patients:
            print("✗ No seizure patients found!")
            return None
        
        # Step 4: Match EEG files
        matched_files = self.match_eeg_files_to_patients(seizure_patients)
        if not matched_files:
            print("✗ No EEG files matched to seizure patients!")
            return None
        
        # Step 5: Process seizure data
        seizure_df = self.process_seizure_data(annotations_df, matched_files)
        if seizure_df.empty:
            print("✗ No seizure data processed!")
            return None
        
        # Step 6: Save results
        output_file = f"{self.dataset_path}/seizure_patients_annotations.csv"
        if self.save_seizure_csv(seizure_df, output_file):
            print("\n" + "=" * 60)
            print("EXTRACTION COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"✓ Processed {len(matched_files)} patients with seizures")
            print(f"✓ Extracted {len(seizure_df)} seizure annotations")
            print(f"✓ Results saved to: {output_file}")
            return seizure_df
        else:
            return None

# Usage example
def main():
    """Main execution function"""
    # Initialize extractor with dataset path
    extractor = MNENeonatalSeizureExtractor(dataset_path="dataset/")
    
    # Run extraction
    seizure_data = extractor.run_extraction()
    
    if seizure_data is not None:
        print("\n" + "=" * 60)
        print("SAMPLE DATA")
        print("=" * 60)
        print(seizure_data.head())
        
        # Display some statistics
        print(f"\nDataset shape: {seizure_data.shape}")
        print(f"Columns: {list(seizure_data.columns)}")
        
        if 'patient_id' in seizure_data.columns:
            print(f"Patients with seizures: {sorted(seizure_data['patient_id'].unique())}")

if __name__ == "__main__":
    main()