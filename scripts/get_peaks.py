import os, sys

if os.uname()[1] == "marmalade.physics.upenn.edu":
    print("I'm on marmalade!")
elif os.uname()[1][:5] == "login":
    print("I'm on perlmutter!")
else:
    sys.exit("I don't know what computer I'm on!")

