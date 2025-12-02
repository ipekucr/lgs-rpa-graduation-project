import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple

class TurkishCommandProcessor:
    def __init__(self, model_name: str = "dbmdz/bert-base-turkish-cased"):
        """Turkish robotics command processor"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)
        
        # Turkish household vocabulary (with morphological variants)
        self.household_objects = {
            "fincan": "cup", "fincanı": "cup", "fincana": "cup", "fincandan": "cup",
            "tabak": "plate", "tabağı": "plate", "tabağa": "plate", "tabaktan": "plate", 
            "çatal": "fork", "çatalı": "fork",
            "kaşık": "spoon", "kaşığı": "spoon",
            "masa": "table", "masadan": "table", "masaya": "table", "masada": "table",
            "sandalye": "chair", "sandalyeyi": "chair",
            "buzdolabı": "refrigerator", "buzdolabına": "refrigerator", "buzdolabından": "refrigerator",
            "mikrodalga": "microwave", "mikrodalgayı": "microwave",
            "çaydanlık": "teapot", "çaydanlığı": "teapot", "çaydanlıktan": "teapot"
        }
        
        # Action verbs
        self.actions = {
            "al": "pick", "koy": "place", "aç": "open", 
            "kapa": "close", "getir": "bring", "götür": "take"
        }
    
    def process_command(self, command: str) -> Dict:
        """Process Turkish command and extract intent + objects"""
        # Tokenize
        inputs = self.tokenizer(command, return_tensors="pt", 
                               padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # [1, 768]
        
        # Extract action and object (simple rule-based for now)
        words = command.lower().split()
        
        detected_action = None
        detected_object = None
        
        for word in words:
            if word in self.actions:
                detected_action = self.actions[word]
            if word in self.household_objects:
                detected_object = self.household_objects[word]
        
        return {
            "command": command,
            "embeddings": embeddings,
            "action": detected_action,
            "object": detected_object,
            "confidence": 0.95 if detected_action and detected_object else 0.5
        }

# Test function
if __name__ == "__main__":
    processor = TurkishCommandProcessor()
    
    test_commands = [
        "Fincanı masadan al",
        "Tabağı buzdolabına koy", 
        "Çaydanlığı getir"
    ]
    
    for cmd in test_commands:
        result = processor.process_command(cmd)
        print(f"Command: {cmd}")
        print(f"Action: {result['action']}, Object: {result['object']}")
        print(f"Confidence: {result['confidence']}")
        print("---")