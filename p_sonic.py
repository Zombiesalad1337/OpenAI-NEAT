import retro
import neat
import numpy as np
import pickle
import cv2


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):
        self.env = retro.make("SonicTheHedgehog-Genesis", "GreenHillZone.Act1")
        self.env.reset()

        ob, _, _, _ = self.env.step(self.env.action_space.sample())

        inx = int(ob.shape[0] / 8)
        iny = int(ob.shape[1] / 8)

        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)

        fitness = 0
        xpos = 0
        xpos_max = 0
        counter = 0

        done = False

        while not done:
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            ob = ob.ravel()
            ob = np.interp(ob, (0, 254), (-1, +1))

            nnOutput = net.activate(ob)

            ob, rew, done, info = self.env.step(nnOutput)
            xpos = info['x']

            if xpos > xpos_max:
                fitness += 1
                counter = 0
                xpos_max = xpos
            else:
                counter += 1

            if counter > 250:
                done = True
            
            if xpos >= info['screen_x_end'] - 2 and xpos > 500: #subtracting 2 from screen_x_end cause sometimes it doesn't trigger the if clause
                fitness += 100000
                done = True
        
        print(fitness)
        return fitness



def eval_genomes(genome, config):
    worker = Worker(genome, config)
    return worker.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)
#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-110')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

pe = neat.ParallelEvaluator(6, eval_genomes)

winner = p.run(pe.evaluate)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

