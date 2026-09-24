from orchestrator.agri_orchestrator import AgriOrchestrator

location = input("Enter location: ")
crop = input("Enter crop: ")

soil = input("Enter soil type (loamy/clay/sandy/black): ").lower()
water = input("Is irrigation available? (yes/no): ").lower()

system = AgriOrchestrator()

result = system.run(location, crop, soil, water)

print(result)