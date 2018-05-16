""" BlueSky plugin template. The text you put here will be visible
    in BlueSky as the description of your plugin. """
# Import the global bluesky objects. Uncomment the ones you need
from bluesky import stack, traf, sim  #, settings, navdb, traf, sim, scr, tools
from bluesky.tools.geo import qdrdist
from bluesky.tools.aero import nm, vcas2tas
from bluesky.tools.misc import degto180
import numpy as np
#from bluesky.sim import simt

### Initialization function of your plugin. Do not change the name of this
### function, as it is the way BlueSky recognises this file as a plugin.
def init_plugin():

    # Addtional initilisation code

    # Configuration parameters
    config = {
        # The name of your plugin
        'plugin_name':     '4D',

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
        'MYFUN': [
            # A short usage string. This will be printed if you type HELP <name> in the BlueSky console
            'MYFUN ON/OFF',

            # A list of the argument types your function accepts. For a description of this, see ...
            '[onoff]',

            # The name of your function in this plugin
            ETA,

            # a longer help text of your function.
            'Print something to the bluesky console based on the flag passed to MYFUN.']
    }

    # init_plugin() should always return these two dicts.
    return config, stackfunctions


### Periodic update functions that are called by the simulation. You can replace
### this by anything, so long as you communicate this in init_plugin

def update():
#    print("Traffic route", traf.ap.route[0].wpname)
#    print(traf.ap.route[0].wptype)
#    print(traf.ap.route[0].wplat)
#    print(traf.ap.route[0].wplon)
#    print(traf.ap.route[0].wpalt)
#    print(traf.ap.route[0].wpspd)
#    print(traf.ap.route[0].wpflyby)
#    print()
    ETA()
#        self.wpname = []
#        self.wptype = []
#        self.wplat  = []
#        self.wplon  = []
#        self.wpalt  = []    # [m] negative value means not specified
#        self.wpspd  = []    # [m/s] negative value means not specified


#        self.wpflyby = []
def preupdate():
    pass

def reset():
    pass

def ETA(acidx):
    # Assuming no wind.
    # Assume complete route is available including orig --> destination

#    print(np.where(traf.ap.route[acidx].wptype))
    # Acceleration value depends on flight phase and aircraft type in BADA
    # The longitudinal acceleration is constant for all airborne flight
    # phases in the bada model.
    ax = traf.perf.acceleration()
    iactwp = traf.ap.route[acidx].iactwp
    nwp = len(traf.ap.route[acidx].wpname)


    # Construct a tables with relevant route parameters before the 4D prediction
    dist2vs =   [0]*nwp
    Vtas =      [0]*nwp
    Vcas =      traf.ap.route[acidx].wpspd.copy()
    dsturn =    [0]*nwp
    qdr =       [0]*nwp
    s =         [0]*nwp
    turndist =  [0]*nwp
    lat =       traf.ap.route[acidx].wplat.copy()
    lon =       traf.ap.route[acidx].wplon.copy()
    altatwp =   traf.ap.route[acidx].wpalt.copy()
    wpialt =    traf.ap.route[acidx].wpialt.copy()

    for wpidx in range(iactwp-1, nwp):
        # If the next waypoint is the active waypoint, use aircraft data
        # Else use previous waypoint data
        if wpidx == iactwp-1:
            Vtas[wpidx] = traf.gs[acidx]       # [m/s]
            Vcas[wpidx] = traf.cas[acidx]
            lat[wpidx] = traf.lat[acidx]    # [deg]
            lon[wpidx] = traf.lon[acidx]    # [deg]
            altatwp[wpidx] = traf.alt[acidx] # [m]

        else:
            lat[wpidx] = traf.ap.route[acidx].wplat[wpidx]          # [deg]
            lon[wpidx] = traf.ap.route[acidx].wplon[wpidx]          #[deg]

            qdr, s[wpidx] = qdrdist(lat[wpidx-1], lon[wpidx-1], lat[wpidx], lon[wpidx]) # [nm]
            s[wpidx] *= nm # [m]

            # check for valid speed, if no speed is given take the speed
            # already in memory from previous waypoint.
            if traf.ap.route[acidx].wpspd[wpidx] < 0:
                Vtas[wpidx] = Vtas[wpidx-1]
                Vcas[wpidx] = Vcas[wpidx-1]
            else:

                Vtas[wpidx] = vcas2tas(traf.ap.route[acidx].wpspd[wpidx],
                          traf.ap.route[acidx].wpalt[wpidx]).item() # [m/s]


            # Compute distance correction for flyby turns.
            if wpidx < nwp - 2:
                #bearing of leg --> Second leg.
                nextqdr, nexts = qdrdist(lat[wpidx], lon[wpidx], lat[wpidx+1], lon[wpidx+1]) #[deg, nm]
                nexts *= nm # [m]
                dist2turn,  turnrad = traf.actwp.calcturn(Vtas[wpidx], traf.bank[acidx], qdr, nextqdr) # [m], [m]
                delhdg = np.abs(degto180(qdr%360-nextqdr%360)) # [deg] Heading change
                distofturn = np.pi*turnrad*delhdg/360 # [m] half of distance on turn radius

                dsturn[wpidx] = dist2turn - distofturn
                turndist[wpidx] = dist2turn


    # Now loop to fill the VNAV constraints. A second loop is necessary
    # Because the complete speed profile and turn profile is required.
    curalt = traf.alt[acidx]
    for wpidx in range(iactwp, nwp):
        # If Next altitude is not the altitude constraint, check if the
        # aircraft should already decent. Otherwise aircraft stays at
        # current altitude. Otherwise expected altitude is filled in.
        if wpidx < wpialt[wpidx]:
        # Compute whether decent should start or speed is still same
            dist2vs[wpidx] = turndist[wpidx] + \
                               np.abs(curalt - traf.ap.route[acidx].wptoalt[wpidx]) / traf.ap.steepness

            # If the dist2vs is smaller everything should be fine
            if traf.ap.route[acidx].wpxtoalt[wpidx] > dist2vs[wpidx]:
                dist2vs[wpidx] = 0
                altatwp[wpidx] = curalt
            else:
                dist2vs[wpidx] = dist2vs[wpidx] - traf.ap.route[acidx].wpxtoalt[wpidx]
                if curalt > traf.ap.route[acidx].wptoalt[wpidx]:
                    sign = -1
                else:
                    sign = 1
                curalt = curalt + dist2vs[wpidx] * traf.ap.steepness * sign
                altatwp[wpidx] = curalt

        # If there is an altitude constraint on the current waypoint compute
        # dist2vs for that waypoint.
        else:
            dist2vs[wpidx] = np.abs(curalt - traf.ap.route[acidx].wptoalt[wpidx]) / traf.ap.steepness
            if dist2vs[wpidx] > traf.ap.route[acidx].wpxtoalt[wpidx]:
                dist2vs[wpidx] = traf.ap.route[acidx].wpxtoalt[wpidx]


    # Start computing the actual 4D trajectory
    t_total = 0
    for wpidx in range(iactwp, nwp-1):
        # If V1 != V2 the aircraft will first immediately accelerate
        # according to the performance model. To compute the time for the
        # leg, constant acceleration is assumed
        Vcas1 = Vcas[wpidx-1]
        Vtas1 = Vtas[wpidx-1]
        Vcas2 = Vcas[wpidx]
        Vtas2 = Vtas[wpidx]
        s_leg = s[wpidx] - dsturn[wpidx-1] - dsturn[wpidx]

        if Vcas1!=Vcas2:
            axsign = 1 + -2*(Vcas1>Vcas2)     # [-]
            t_acc = (Vcas2-Vcas1)/ax * axsign    # [s]
            s_ax = t_acc * (Vtas2+Vtas1)/2         # [m] Constant acceleration
        else:
            s_ax = 0 # [m]
            t_acc = 0

        s_nom = s_leg - s_ax - dist2vs[wpidx] # [m] Distance of normal flight
        # Check if acceleration distance and start of descent overlap
        if s_nom < 0:
            t_vs = 0
            t_nom = 0
            if dist2vs[wpidx] <= s_leg:
                s1 = 0
                s2 = s_ax
                s3 = s_leg - s_ax
            else:
                s1 = s_leg - dist2vs[wpidx]
                s2 = dist2vs[wpidx] + s_ax - s_leg
                s3 = s_leg - s_ax


            t1 = abc(0.5*vcas2tas(ax, altatwp[wpidx-1]), Vtas1, s1)
            Vcas_vs = Vcas1 - t1 * ax
            t2 = (Vcas2-Vcas_vs)/ax
            alt2 = altatwp[wpidx-1] - s2 * traf.ap.steepness
            Vtas_vs = vcas2tas(Vcas_vs, alt2)
            t3 = s3/((Vtas_vs + Vtas2)/2)
            t_leg = t1 + t2 + t3

        else:
            Vtas_nom = vcas2tas(Vcas2, altatwp[wpidx-1])
            t_nom = s_nom/Vtas_nom

            #Assume same speed for now.
            t_vs = dist2vs[wpidx]/(Vtas_nom + Vtas2)/2

            if dist2vs[wpidx] < 1:
                t_vs = 0

            t_leg = t_nom + t_acc + t_vs
        t_total+=t_leg


    # Debug print statements
#    print('wptype ', traf.ap.route[acidx].wptype[traf.ap.route[acidx].iactwp])
#    print('DEBUG')
#    print('Vtas ', Vtas)
#    print('Vcas ', Vcas)
#    print('dsturn ', dsturn)
#    print('s ', s)
#    print('turndist ', turndist)
#    print('lat ', lat)
#    print('lon ', lon)
#    print('altatwp ', altatwp)
#    print('wpialt ', traf.ap.route[acidx].wpialt)
#    print('wptoalt ', traf.ap.route[acidx].wptoalt)
#    print('wpxtoalt ', traf.ap.route[acidx].wpxtoalt)
#    print('altatwp ', altatwp)
#    print('dist2vs ', dist2vs)
#    print('eta', sim.simt + t_total)
#    print(' ')
#    print(t_total)
    return t_total[0] + sim.simt


def abc(a, b, c):
    D = b*b - 4*a*c
    x1 =( -b - D**0.5 )/ (2*a)
    x2 =( -b + D**0.5 )/ (2*a)
    x = x1*x1>0 + x2*x2>0
    return x