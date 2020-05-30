import retro
import numpy as np
import cv2 
import neat
import pickle
import time
import simpleaudio as sa
from threading import Thread

class sound():
    def play(self, array, fs):
        sa.play_buffer(array, 2, 2, 44100)
mysound = sound()

env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')


p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

with open('winner.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)

ob = env.reset()

inx, iny, _ = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0
xpos = 0
xpos_max = 0

env.render()
time.sleep(2)
done = False

while not done:
    env.render()
    frame += 1

    audio = env.em.get_audio()
    audio_rate = env.em.get_audio_rate()
    thread = Thread(target=mysound.play, args=(audio, audio_rate,))
    thread.start()

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx, iny))
    ob = ob.ravel()
    ob = np.interp(ob, (0, 254), (-1, +1))
    nnOutput = net.activate(ob)

    ob, rew, done, info = env.step(nnOutput)

    time.sleep(0.004)
