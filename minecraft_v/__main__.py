from minecraft_v.progress import phase

with phase("[load] starting"):
    from minecraft_v.main import main

main()
