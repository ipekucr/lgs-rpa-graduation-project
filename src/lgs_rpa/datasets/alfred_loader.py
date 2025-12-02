import json
import os
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

class ALFREDDataLoader:
    def __init__(self, data_dir: str = "data/alfred_repo/data/json_2.1.0"):
        self.data_dir = Path(data_dir)
        self.valid_seen_dir = self.data_dir / "valid_seen"
        
        # Discrete pose adjustment actions we're interested in
        self.pose_actions = [
            "MoveAhead", "MoveBack", "MoveLeft", "MoveRight",
            "RotateLeft", "RotateRight", "LookUp", "LookDown"
        ]
        
    def extract_pose_adjustment_cases(self) -> List[Dict]:
        """Extract discrete pose adjustment cases for CVAE training"""
        pose_cases = []
        
        # Process only first 20 files for testing
        file_count = 0
        for task_dir in self.valid_seen_dir.iterdir():
            if task_dir.is_dir() and file_count < 20:
                for trial_dir in task_dir.iterdir():
                    if trial_dir.is_dir():
                        traj_file = trial_dir / "traj_data.json"
                        if traj_file.exists():
                            cases = self._process_trajectory_file(traj_file)
                            pose_cases.extend(cases)
                            file_count += 1
                            if file_count >= 20:
                                break
        
        print(f"Extracted {len(pose_cases)} pose adjustment cases from {file_count} files")
        return pose_cases
    
    def _process_trajectory_file(self, traj_file: Path) -> List[Dict]:
        """Process single trajectory file and extract pose cases"""
        try:
            with open(traj_file) as f:
                data = json.load(f)
            
            cases = []
            
            # Check if plan exists
            if 'plan' not in data or 'low_actions' not in data['plan']:
                return cases
            
            low_actions = data['plan']['low_actions']
            
            for i, action in enumerate(low_actions):
                # DÜZELTME: api_action içindeki action'ı al
                action_name = action.get('api_action', {}).get('action', 'UNKNOWN')
                
                if action_name in self.pose_actions:
                    # Extract context
                    context = self._extract_context(data, i)
                    
                    case = {
                        'discrete_action': action_name,
                        'trajectory_id': str(traj_file.parent.name),
                        'action_index': i,
                        'context': context,
                        'coordinate': action.get('coordinate', [0, 0, 0]),
                        'rotation': action.get('rotation', 0)
                    }
                    cases.append(case)
            
            return cases
            
        except Exception as e:
            print(f"Error processing {traj_file}: {e}")
            return []
    
    def _extract_context(self, data: Dict, action_index: int) -> Dict:
        """Extract context information for the action"""
        context = {}
        
        # Get task description
        if 'turk_annotations' in data and 'anns' in data['turk_annotations']:
            anns = data['turk_annotations']['anns'][0]
            context['task_description'] = anns.get('task_desc', '')
            context['high_level_desc'] = anns.get('high_descs', [''])[0] if anns.get('high_descs') else ''
        
        # Get previous actions
        if action_index > 0:
            prev_actions = data['plan']['low_actions'][:action_index]
            prev_action_names = []
            for prev_action in prev_actions[-3:]:
                prev_name = prev_action.get('api_action', {}).get('action', 'UNKNOWN')
                prev_action_names.append(prev_name)
            context['previous_actions'] = prev_action_names
        else:
            context['previous_actions'] = []
        
        # Get scene information
        if 'scene' in data:
            context['scene_num'] = data['scene'].get('scene_num', 0)
        
        return context

# Test function
if __name__ == "__main__":
    loader = ALFREDDataLoader()
    pose_cases = loader.extract_pose_adjustment_cases()
    
    if pose_cases:
        print(f"\nFirst pose case example:")
        print(f"Discrete action: {pose_cases[0]['discrete_action']}")
        print(f"Context: {pose_cases[0]['context']['task_description'][:100]}...")
        print(f"Previous actions: {pose_cases[0]['context']['previous_actions']}")
        
        # Count action types
        action_counts = {}
        for case in pose_cases:
            action = case['discrete_action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        print(f"\nAction distribution:")
        for action, count in sorted(action_counts.items()):
            print(f"  {action}: {count}")
    else:
        print("No pose adjustment cases found!")