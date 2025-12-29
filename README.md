# MemCtrl: Task-Aware Memory Management for Long-Context LLMs

**Enable unlimited conversation length on commodity hardware (8GB GPU)**

## Features

- 🧠 **Task-Aware Policies**: Learns what to remember for medical, code, writing tasks
- 🎮 **User Control**: Pin, forget, or mark information as temporary
- 💾 **Cross-Session Persistence**: Never forgets important information
- 🖥️ **Commodity Hardware**: Works on 8GB GPU (not 80GB)
- 🔒 **Privacy-First**: All data stored locally

## Installation
```bash
git clone https://github.com/yourusername/memctrl.git
cd memctrl
pip install -e .
```

## Quick Start

### Python API
```python
from memctrl import MemoryController

controller = MemoryController()
response = controller.chat("Help me debug this code", user_id="kamala")
controller.pin("I prefer TypeScript", user_id="kamala")
```

### CLI
```bash
memctrl chat "Hello" --user kamala
memctrl pin "I'm allergic to penicillin" --user kamala
memctrl show --user kamala
```

### Web UI
```bash
memctrl start
# Opens at http://localhost:7860
```

## Citation

If you use MemCtrl in your research, please cite:
```bibtex
@inproceedings{memctrl2025,
  title={MemCtrl: Task-Aware Memory Management with User Control for Long-Context Language Models},
  author={Your Name},
  booktitle={ACL},
  year={2025}
}
```

## License

MIT License