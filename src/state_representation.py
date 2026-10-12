import numpy as np
from typing import Tuple, Any, Optional

# --- 1. Visual State Representation ---

class VisualState:
    def __init__(self, image_array: np.ndarray):
        if not isinstance(image_array, np.ndarray):
            raise TypeError("VisualState requires a NumPy array input.")
        self.image = image_array

    def get_tensor(self) -> np.ndarray:
        return self.image

# --- 2. Graph/Map State Representation ---

class GraphState:
    def __init__(self, coordinates: np.ndarray, connections: np.ndarray = None):
        self.coordinates = coordinates
        self.connections = connections if connections is not None else np.zeros((len(coordinates), len(coordinates)))

    def get_features(self) -> np.ndarray:
        return np.concatenate([self.coordinates.flatten(), self.connections.flatten()[:self.coordinates.shape[0] * 5]])

# --- 3. Text/Goal Specification ---

class TextState:
    def __init__(self, instruction: str, embedding: np.ndarray = None):
        self.instruction = instruction
        self.embedding = embedding if embedding is not None else np.random.rand(512).astype(np.float32)

# --- 4. Code State (for code artifacts and prompts) ---

class CodeState:
    def __init__(self, initial_prompt: str, current_code_artifact: str, code_embedding: np.ndarray):
        self.initial_prompt = initial_prompt
        self.current_code_artifact = current_code_artifact
        self.code_embedding = code_embedding

    def get_features(self) -> np.ndarray:
        return self.code_embedding

# --- 5. Execution State (feedback from execution) ---

class ExecutionState:
    def __init__(self, error_log: str, execution_trace: str, updated_graph_features: np.ndarray):
        self.error_log = error_log
        self.execution_trace = execution_trace
        self.updated_graph_features = updated_graph_features

    def get_features(self) -> np.ndarray:
        return self.updated_graph_features

# --- 6. Combined State ---

class AgentState:
    def __init__(self, visual: VisualState, graph: Optional[GraphState] = None, text: Optional[TextState] = None, code: Optional[CodeState] = None, exec_state: Optional[ExecutionState] = None):
        self.visual = visual
        self.graph = graph
        self.text = text
        self.code = code
        self.exec_state = exec_state
        
        components = [visual.get_tensor().flatten()]
        if graph is not None:
            components.append(graph.get_features())
        if text is not None:
            components.append(text.embedding)
        if code is not None:
            components.append(code.get_features())
        if exec_state is not None:
            components.append(exec_state.get_features())
        
        self.state_vector = np.concatenate(components).astype(np.float32)

    def get_combined_state(self) -> np.ndarray:
        return self.state_vector