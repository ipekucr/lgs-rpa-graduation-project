import torch
import numpy as np
import cv2
from models.turkish_processor import TurkishCommandProcessor
from models.perception_module import PerceptionModule

class RobotIntegrationTest:
    def __init__(self):
        self.turkish_processor = TurkishCommandProcessor()
        self.perception = PerceptionModule(num_classes=21)
        self.perception.to(self.perception.device)  # Device'a taşı
        
    def process_command_with_vision(self, command: str, image: np.ndarray):
        """Turkish command + vision integration"""
        # Process Turkish command
        language_result = self.turkish_processor.process_command(command)
        
        # Process image
        image_tensor = self.perception.preprocess_image(image)
        with torch.no_grad():
            segmentation = self.perception(image_tensor)
        
        # Find target object in image
        target_object = language_result['object']
        object_found = self.find_object_in_segmentation(target_object, segmentation)
        
        return {
            'command': command,
            'action': language_result['action'],
            'target_object': target_object,
            'object_found_in_image': object_found,
            'segmentation_shape': segmentation.shape
        }
    
    def find_object_in_segmentation(self, target_object: str, segmentation: torch.Tensor):
        """Simple object detection in segmentation map"""
        object_to_class = {
            'cup': 3, 'table': 1, 'chair': 2, 'plate': 4,
            'refrigerator': 7, 'microwave': 8, 'teapot': 11
        }
        
        if target_object in object_to_class:
            class_idx = object_to_class[target_object]
            predicted_classes = segmentation.argmax(dim=1)
            object_pixels = (predicted_classes == class_idx).sum().item()
            return object_pixels > 100
        
        return False

if __name__ == "__main__":
    integration = RobotIntegrationTest()
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    test_commands = [
        "Fincanı masadan al",
        "Tabağı buzdolabına koy"
    ]
    
    for cmd in test_commands:
        result = integration.process_command_with_vision(cmd, test_image)
        print(f"Command: {result['command']}")
        print(f"Action: {result['action']}")
        print(f"Target: {result['target_object']}")
        print(f"Found in image: {result['object_found_in_image']}")
        print("---")