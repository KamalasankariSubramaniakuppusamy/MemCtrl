from memctrl import MemoryController

controller = MemoryController(user_id="kamala")
response = controller.chat("Help me with Python")
print(response)

controller.pin("I prefer TypeScript", note="Language preference")
memory = controller.show_memory()
print(memory)

stats = controller.get_stats()
print(stats)