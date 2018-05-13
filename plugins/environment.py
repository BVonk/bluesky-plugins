""" Plugin to resolve conflicts """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, sim  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools.geo import kwikdist, qdrpos, qdrdist
from bluesky.tools.misc import degto180
from bluesky import traf
from bluesky import navdb
from bluesky.tools.aero import nm, g0
from vierd import ETA

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():
    # Addtional initilisation code
    # Pan to waypoint with fixed zoom level and create a random aircraft.
    # Configuration parameters
    global env, agent, eventmanager, state_size
    state_size = 3
    env = Env()
    agent = DQNAgent(state_size,3)
    eventmanager = Eventmanager()

    config = {
        # The name of your plugin
        'plugin_name':     'env',

        # The type of this plugin. For now, only simulation plugins are possible.
        'plugin_type':     'sim',

        # Update interval in seconds. By default, your plugin's update function(s)
        # are called every timestep of the simulation. If your plugin needs less
        # frequent updates provide an update interval.
        'update_interval': 1,

        # The update function is called after traffic is updated. Use this if you
        # want to do things as a result of what happens in traffic. If you need to
        # something before traffic is updated please use preupdate.
        'update':          update,

        # The preupdate function is called before traffic is updated. Use this
        # function to provide settings that need to be used by traffic in the current
        # timestep. Examples are ASAS, which can give autopilot commands to resolve
        # a conflict.
        'preupdate':       preupdate,

        # If your plugin has a state, you will probably need a reset function to
        # clear the state in between simulations.
        'reset':         reset
        }

    stackfunctions = {
        # The command name for your function
#        "ENV_STEP": [
#            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
#            'Step the environment',
#            '',
#            env.step,
#
#            # a longer help text of your function.
#            'Print something to the bluesky console based on the flag passed to MYFUN.'],
#
#        "ENV_ACT": [
#            'hdg',
#            'hdg',
#            env.act,
#            'Select a heading'],
#
#        "ENV_RESET": [
#            'Reset the training environment',
#            '',
#            env.reset,
#            'Reset the training environment']
            }


    # init_plugin() should always return these two dicts.
    return config, stackfunctions




### Other functions of your plugin
#def conflict_resolution():
#    print(traf.asas.conflist_now)
#    for conf in traf.asas.conflist_now:
#        intruder = conf.split(' ')[-1]
#        stack.stack(intruder + ' DEL')
#        stack.stack('ECHO Deleted intruder '+intruder)

def update():
    train() if train_phase else test()

def train():
    eventmanager.update()


    for i in eventmanager.events:
        if env.actnum == 0:
            agent.sta = ETA(agent.acidx) + random.random() * 100
#            print('STA ', agent.sta)
        next_state, reward, done, prev_state = env.step()
#        print('state ', next_state)
#        print('reward ', reward)
        next_state = np.reshape(next_state, [1, agent.state_size])
        if env.actnum>0:
            agent.remember(prev_state, agent.action, reward, next_state, done)
        if len(agent.memory) > agent.batch_size:
            agent.replay(agent.batch_size)
        if not done:
            agent.act(next_state)

def test():
    eventmanager.update()


    for i in eventmanager.events:
        if env.actnum == 0:
            agent.sta = ETA(agent.acidx) + random.random() * 100
#            print('STA ', agent.sta)
        next_state, reward, done, prev_state = env.step()

        if not done:
            agent.act_test(next_state)



def preupdate():
#    if len(traf.id) !=0:
#        agent.act(env.state)
#        env.act(agent.action)
    pass


def reset():
    pass


class Env:
    def __init__(self):
        self.acidx = 0
        self.reward = 0
        self.done = False
        self.state = np.ones((1, state_size))
        self.actnum = 0
        self.ep = 0
        self.fname = './output/log.csv'

        if not os.path.file_exists(self.fname):
            f = open(self.fname, 'w')
            f.write("Episode;reward;dist;hdg;t;epsilon\n")
            f.close()

        self.reset()

    def step(self):
#        print("Step", self.actnum)
        self.actnum += 1
        prev_state = self.state

        # Update state
        qdr, dist = qdrdist(traf.lat[self.acidx], traf.lon[self.acidx],
                            traf.ap.route[self.acidx].wplat[-2],
                            traf.ap.route[self.acidx].wplon[-2])
        t = agent.sta - sim.simt
        hdg_ref = abs(degto180(qdr - traf.hdg[agent.acidx]))
#        self.state = np.array([dist, t, hdg_ref,
#                               traf.tas[agent.acidx]])
        self.state = np.array([dist, t, hdg_ref])
        # Check episode termination
        if dist<1 or t<-60:
            self.done = True
            env.reset()

        reward = self.gen_reward()

        return self.state, reward, self.done, prev_state


    def gen_reward(self):
        dist = self.state[0]
        t = self.state[1]
        hdg = self.state[2]
        hdg_ref = 60.

        a_dist = -1
        a_t = -1.
        a_hdg = -0.07
        dist_rew = 2 + a_dist * dist

        if self.done:
            t_rew = 5 + a_t * abs(t)
            hdg_rew = a_hdg * abs(degto180(hdg_ref - hdg))

        else:
            t_rew = 0
            hdg_rew = 0

        self.reward = dist_rew + t_rew + hdg_rew
        return self.reward


    def reset(self):
        if self.ep%5 == 0 and self.ep!=0:
            agent.save("./output/model{0:05}.hdf5".format(self.ep))
            print("Saving model after {} episodes".format(self.ep))

        if self.ep>0:
            self.log()
        stack.stack('open ./scenario/4d.SCN')
        self.actnum = 0
        self.ep += 1
        self.done=False
        print("Episode ", self.ep)

    def log(self):
        dist = self.state[0]
        t = self.state[1]
        hdg = self.state[2]
        hdg_ref = 60.
        hdg_diff = (degto180(hdg_ref - hdg))
        f = open(self.fname, 'a')
        self.f.write("{};{};{};{};{}\n".format(self.ep, self.reward, dist, hdg_diff, t, agent.epsilon))
        f.close()

    def act(self, action):
        # Set new heading reference of the aircraft
        stack.stack(traf.id[self.acidx] + ' HDG ' + str(action))


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.train = True
        self.fname = './output/model00190.HDF5' #'model00005.hdf5'
        self.acidx = 0
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9985
        self.learning_rate = 0.001
        self.batch_size = 32
        self.done = False
        self.sta = 0
        self.action = 0
        self.actions = [self.act1, self.act4, self.act5]
        self.replaysteps = 0
        self.model = self._build_model()

        if self.train and not self.fname=='':
            self.load(self.fname)
#
        elif not self.train:
            self.load(self.fname)

        self.targetmodel = self.model
        print(self.model.summary())

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(52, input_dim=self.state_size, activation='relu'))
        model.add(Dense(52, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer='rmsprop')
#                      optimizer=Adam(lr=self.learning_rate))
        print(model.summary())
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act1(self):
        # Compute 15 degree turn left waypoint.
        # Use current speed to compute waypoint.

        dqdr = 15
        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]
        turnrad = traf.tas[self.acidx]**2 / (np.maximum(0.01, np.tan(traf.bank[self.acidx])) * g0) # [m]

        #Turn right so add bearing
#        qdr = traf.qdr[self.acidx] + 90

        latR, lonR = qdrpos(latA, lonA, traf.hdg[self.acidx] + 90, turnrad/nm) # [deg, deg]
        # Rotate vector
        latB, lonB = qdrpos(latR, lonR, traf.hdg[self.acidx] - 90 + dqdr, turnrad/nm) # [deg, deg]
        cmd = "{} BEFORE {} ADDWPT '{},{}'".format(traf.id[self.acidx], traf.ap.route[0].wpname[-2], latB, lonB)
        stack.stack(cmd)


    def act2(self):
        pass


    def act3(self):
        pass


    def act4(self):
        # Compute 15 degree turn left waypoint.
        # Use current speed to compute waypoint.

        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]

        latB, lonB = qdrpos(latA, lonA, traf.hdg[self.acidx], 0.25)
        cmd = "{} BEFORE {} ADDWPT '{},{}'".format(traf.id[self.acidx], traf.ap.route[0].wpname[-2], latB, lonB)
        stack.stack(cmd)


    def act5(self):
        # Compute 15 degree turn left waypoint.
        # Use current speed to compute waypoint.
        dqdr = 15
        latA = traf.lat[self.acidx]
        lonA = traf.lon[self.acidx]
        turnrad = traf.tas[self.acidx]**2 / (np.maximum(0.01, np.tan(traf.bank[self.acidx])) * g0) # [m]

        #Turn right so add bearing
#        qdr = traf.qdr[self.acidx] + 90

        latR, lonR = qdrpos(latA, lonA, traf.hdg[self.acidx] - 90, turnrad/nm) # [deg, deg]
        # Rotate vector
        latB, lonB = qdrpos(latR, lonR, traf.hdg[self.acidx] + 90 - dqdr, turnrad/nm) # [deg, deg]
        cmd = "{} BEFORE {} ADDWPT '{},{}'".format(traf.id[self.acidx], traf.ap.route[0].wpname[-2], latB, lonB)
        stack.stack(cmd)


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            agent.action = random.choice(np.arange(0, self.action_size))

        else:
            act_values = self.model.predict(env.state.reshape((1,agent.state_size)))
            self.action = np.argmax(act_values[0])
#            print('Act ', act_values[0])

        # Pick the action with the highest Q-value

        self.actions[self.action]()

    def act_test(self, state):
        act_values = self.model.predict(env.state.reshape((1,agent.state_size)))
        self.action = np.argmax(act_values[0])

        self.actions[self.action]()

    def replay(self, batch_size):
        c = 100
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:

            # If done, make the target reward
            target = reward

            if not done:
                # Predict the future discounted reward
                target = reward + self.gamma * np.amax(self.model.predict(next_state.reshape((1,agent.state_size)))[0])

            # make the agent to approximately map
            # the current state to future discounted reward
            # We'll call that target_f
            target_f = self.targetmodel.predict(state.reshape((1,self.state_size)))
            target_f[0][action] = target

            # Train the Neural Net with the state and target_f
#            print('target', target)
#            print(st)
            self.model.fit(state.reshape((1,agent.state_size)), target_f, epochs=1, verbose=0)

        if self.replaysteps%c == 0:
            self.targetmodel = self.model


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.replaysteps += 1


    def load(self, name):
        print("Loading weights from {}".format(self.fname))
        self.model.load_weights(name)
        env.ep = int(self.fname[-10:-5])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay**env.ep)


    def save(self, name):
        self.model.save_weights(name)


class Eventmanager():
    def __init__(self):
        self.eventidx = []


    def update(self):
        self.events = []
        for acidx in range(traf.ntraf):
            if traf.ap.route[acidx].iactwp == len(traf.ap.route[acidx].wptype) - 2:
                self.events.append(acidx)
