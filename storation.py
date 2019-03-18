import numpy as np


def store_transition(self, s, a, r, s_):
    if not hasattr(self, 'memory_counter'):
        self.memory_counter = 0

    # 记录一条 [s, a, r, s_] 记录
    transition = np.hstack((s, [a, r], s_))

    # 总 memory 大小是固定的, 如果超出总大小, 旧 memory 就被新 memory 替换
    index = self.memory_counter % self.memory_size
    self.memory[index, :] = transition  # 替换过程

    self.memory_counter += 1