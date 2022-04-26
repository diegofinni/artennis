import shutil

total, used, free = shutil.disk_usage(__file__)
print(total, used, free)
