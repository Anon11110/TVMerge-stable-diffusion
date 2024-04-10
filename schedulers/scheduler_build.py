import json
from typing import Callable, ClassVar, Dict, List, Type

import numpy as np
import tvm
from tvm import relax
from tvm.script import relax as R


class BaseScheduler:
    constants_file_name: ClassVar[str]

    @staticmethod
    def scheduler_steps() -> tvm.IRModule:
        raise NotImplementedError()
    
    @staticmethod
    def list_step_functions() -> List[str]:
        raise NotImplementedError()
    
    @staticmethod
    def calculate_constants() -> Dict[str, List[tvm.nd.NDArray]]:
        raise NotImplementedError()
    

schedulers_build: List[Type[BaseScheduler]]