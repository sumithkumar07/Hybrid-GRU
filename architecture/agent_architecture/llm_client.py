"""
hybrid_gru Brain LLM Client Wrapper
Replaces OpenAI API calls with native C++ hybrid_gru inference
"""

import os
import ctypes
import numpy as np
from typing import Optional, Dict, Any, List

# Define the path to the hybrid_gru DLL/SO in the backend root
# Final Production Path (Root of the infrastructure)
LIB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "hybrid_gru.dll"))

class hybrid_gruEngine:
    """Singleton wrapper for the 100M Master Brain (Hive Engine v13.1)"""
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def __init__(self):
        self._fragment_registry = {}
        try:
            self.lib = ctypes.CDLL(LIB_PATH)
            
            # Function Signatures (Phase 9: Swarm & Memory Anchors)
            self.lib.hybrid_gru_init_master.argtypes = []
            self.lib.hybrid_gru_init_master.restype = ctypes.c_void_p
            self.lib.hybrid_gru_init_agent.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int]
            self.lib.hybrid_gru_init_agent.restype = ctypes.c_void_p
            
            self.lib.hybrid_gru_init_fragment.argtypes = [ctypes.c_char_p]
            self.lib.hybrid_gru_init_fragment.restype = ctypes.c_void_p
            self.lib.hybrid_gru_agent_set_fragment.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            
            self.lib.hybrid_gru_agent_save_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.lib.hybrid_gru_agent_load_state.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            
            self.lib.hybrid_gru_agent_observe.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.lib.hybrid_gru_agent_act.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_double]
            self.lib.hybrid_gru_agent_act.restype = ctypes.c_char_p
            self.lib.hybrid_gru_free_agent.argtypes = [ctypes.c_void_p]
            self.lib.hybrid_gru_load_compact.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.lib.hybrid_gru_save_compact.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            
            # Neural Injection (Phase 7: Distillation)
            self.lib.hybrid_gru_train_step_distill.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_float]
            self.lib.hybrid_gru_train_distill_bulk.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_float]

            # Fragmentation Persistence
            self.lib.hybrid_gru_get_fragment_bias.restype = ctypes.POINTER(ctypes.c_float)
            
            # Tokenization (Phase 7.2: High-Fidelity Mapping)
            self.lib.hybrid_gru_tokenize.argtypes = [ctypes.c_char_p]
            self.lib.hybrid_gru_tokenize.restype = ctypes.c_int
            self.lib.hybrid_gru_detokenize.argtypes = [ctypes.c_int]
            self.lib.hybrid_gru_detokenize.restype = ctypes.c_char_p
            
            # Allocation
            self.master_brain = self.lib.hybrid_gru_init_master()
            if os.path.exists("master_brain_compact.bin"):
                self.lib.hybrid_gru_load_compact(self.master_brain, b"master_brain_compact.bin")
                print(f"[hybrid_gru] Hive Engine Initialized (16MB Compact).")
        except Exception as e:
            print(f"[hybrid_gru ERROR] Hive Engine failed: {e}")
            self.lib = None

class hybrid_gruSwarm:
    """The Swarm Director: Orchestrates collaboration between agent fragments"""
    def __init__(self):
        self.engine = hybrid_gruEngine.get_instance()
        self.agent_ptr = self.engine.lib.hybrid_gru_init_agent(b"SwarmDirector", self.engine.master_brain, 42)
        self.specialists = {}

    def spawn_specialist(self, name: str, bias_data: Optional[np.ndarray] = None):
        """Register a new specialist fragment in the swarm"""
        frag_ptr = self.engine.lib.hybrid_gru_init_fragment(name.encode('utf-8'))
        # If we have a soul file (personality + memory), restore it
        soul_file = f"{name}.soul.npy"
        if os.path.exists(soul_file):
            state = np.load(soul_file, allow_pickle=True).item()
            # Restore bias
            bias_ptr = self.engine.lib.hybrid_gru_get_fragment_bias(frag_ptr)
            np.copyto(np.ctypeslib.as_array(bias_ptr, shape=(1024,)), state['bias'])
            # Restore Memory Anchor (H-state)
            # This is handled manually by copying into H-state before saving
            ctypes.memmove(
                ctypes.c_void_p(frag_ptr + 4096 + 40), # Estimate offset past ID and Bias
                state['h'].ctypes.data, 
                8*1024
            )
        self.specialists[name] = frag_ptr
        print(f"[SWARM] Specialist Ready: {name}")

    def collaborate(self, task: str, sequence: List[str]) -> Dict[str, str]:
        """Run a task through a relay of specialists (True Agentic Swarm)"""
        results = {}
        current_input = task
        
        for name in sequence:
            if name not in self.specialists:
                self.spawn_specialist(name)
            
            # Activate specialist (Auto-restores Memory Anchor)
            self.engine.lib.hybrid_gru_agent_set_fragment(self.agent_ptr, self.specialists[name])
            
            # Observe and Act
            prompt = f"Relay from previous agent: {current_input}\nTask: {task}"
            self.engine.lib.hybrid_gru_agent_observe(self.agent_ptr, prompt.encode('utf-8'))
            
            raw_res = self.engine.lib.hybrid_gru_agent_act(self.agent_ptr, 100, 0.7)
            response = raw_res.decode('utf-8', errors='ignore').strip().replace('Ġ', ' ')
            
            results[name] = response
            current_input = response
            print(f"[SWARM] {name} finished turn.")
            
        return results

class LLMClient:
    """Lite Client for direct interacting with the Swarm Engine"""
    def __init__(self):
        self.engine = hybrid_gruEngine.get_instance()
        self.swarm = hybrid_gruSwarm()

    def chat(self, messages: List[Dict[str, str]], personality: str = "Generalist", **kwargs) -> str:
        """Execute a single specialist chat with memory preservation"""
        if personality not in self.swarm.specialists:
            self.swarm.spawn_specialist(personality)
            
        # Switch Personality (Persistence handled by C++ auto-save)
        self.engine.lib.hybrid_gru_agent_set_fragment(self.swarm.agent_ptr, self.swarm.specialists[personality])
        
        context = "\n".join([f"[{m['role'].upper()}]: {m['content']}" for m in messages])
        self.engine.lib.hybrid_gru_agent_observe(self.swarm.agent_ptr, context.encode('utf-8'))
        
        raw_res = self.engine.lib.hybrid_gru_agent_act(self.swarm.agent_ptr, 200, 0.7)
        return raw_res.decode('utf-8', errors='ignore').strip().replace('Ġ', ' ')

    def train_step(self, input_token: int, teacher_probs: np.ndarray, lr: float = 0.001):
        """Perform a single weight update from a teacher token"""
        if not self.engine.lib: return
        probs_ptr = teacher_probs.astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.engine.lib.hybrid_gru_train_step_distill(self.swarm.agent_ptr, input_token, probs_ptr, lr)

    def train_bulk(self, tokens: List[int], lr: float = 0.001):
        """Turbo: Perform a batch weight update for a sequence of tokens in C++"""
        if not self.engine.lib: return
        n = len(tokens)
        tokens_arr = (ctypes.c_int * n)(*tokens)
        self.engine.lib.hybrid_gru_train_distill_bulk(self.swarm.agent_ptr, tokens_arr, n, lr)
