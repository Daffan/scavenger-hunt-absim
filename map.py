###############################################################
# The map is represented as the same .dat file as Jennifer's 
# implementaton. Some methods here are directly copied from 
# scavenger_hunt_api
###############################################################
import random
import math
import numpy as np, numpy.random


######################## Copied from scavenger_hunt_api ############################

def sample(ls):
    item = random.choice(ls)
    ls.remove(item)
    return item

# Distribute probabilities uniformly with Dirichlet distribution
def distribute(count):
    return np.random.dirichlet(np.ones(count),size=1)[0]


def generate(fname, nodes_range, cost_range, objects_range, occurrences_range):
    out = open(fname, "w")

    # Map
    out.write("[map]\n")
    start_marked = False
    c_nodes = random.randint(nodes_range[0], nodes_range[1])
    nodes = []
    points = []
    first = True
    second = True

    for i in range(c_nodes):
        node = "l%s" % i

        # Generate a random point
        # if first: #Robot in middle
        #     point = (random.randrange(200, 300, 1), random.randrange(200, 300, 1))
        #     first = False
        # elif second: # One point on one side
        #     point = (random.randrange(100, 150, 1), random.randrange(100, 150, 1))
        #     second = False
        # else: # All other points on different side
        #     point = (random.randrange(500, 700, 1), random.randrange(500, 700, 1))

        # Original
        point = (random.randrange(cost_range[0], cost_range[1], 1), random.randrange(cost_range[0], cost_range[1], 1))

        index = 0

        # Get distances between each point
        for point2 in points:
            cost = math.sqrt((point2[0] - point[0])**2 + (point2[1] - point[1])**2)
            start = "*" if not start_marked else ""
            start_marked = True
            out.write("%s%s l%s %s\n" %(node, start, index, cost))
            index = index + 1

        nodes.append(node)
        points.append(point)

    # Distribution
    out.write("\n[distr]\n")
    c_objs = random.randint(objects_range[0], objects_range[1])
    for obj in range(c_objs):
        out.write("o%s " % obj)
        occurrences = random.randint(occurrences_range[0], occurrences_range[1])
        locs = nodes.copy()
        locs.remove("l1") # avoid object at starting position
        p = distribute(occurrences)
        for j in range(len(p)):
            out.write("%s %s " % (sample(locs), p[j]))
        out.write("\n")
    out.close()

def parse_world(fname):
    """Parses a scavenger hunt world from a datfile.
    Parameters
    ----------
    fname : str
        source file name
    Returns
    -------
    
    """
    src = open(fname, "r")
    sec = None
    start_loc = None
    conns = []  # 3-tuples (from, to, cost)
    nodes = {}  # User-specified loc names -> integer IDs
    distrs = []
    objs = []
    node_count = 0

    for line in src.readlines():
        # Blank lines and comments
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            continue

        # Section header line
        if line[0] == '[':
            sec = line[1:line.find(']')]
            continue

        # Map section
        if sec == "map":
            args = line.split()
            assert len(args) == 3
            n_from, n_to = args[0], args[1]

            # Parse for starting location
            if '*' in n_from:
                n_from = n_from.replace('*', '')
                if start_loc is None:
                    start_loc = n_from
            elif '*' in n_to:
                n_to = n_to.replace('*', '')
                if start_loc is None:
                    start_loc = n_to

            cost = float(args[2])
            if n_from not in nodes:
                nodes[n_from] = node_count
                node_count += 1
            if n_to not in nodes:
                nodes[n_to] = node_count
                node_count += 1
            conns.append((n_from, n_to, cost))
        
        # Distribution section
        elif sec == "distr":
            distr = np.zeros(node_count)
            args = line.split()
            assert len(args) > 2
            obj = args[0]
            ind = 1

            if obj not in objs:
                objs.append(obj)

            while ind < len(args):
                prob_ind = ind
                while args[prob_ind] in nodes:
                    loc = nodes[args[prob_ind]]
                    prob_ind += 1
                prob_arg = args[prob_ind]
                if '/' in prob_arg:
                    frac = prob_arg.split('/')
                    prob = float(frac[0]) / float(frac[1])
                else:
                    prob = float(prob_arg)
                distr[loc] = prob
                ind = prob_ind + 1

            distrs.append(distr)
        else:
            assert False

    src.close()

    graph = np.zeros((node_count, node_count))
    for c in conns:
        n1 = nodes[c[0]]
        n2 = nodes[c[1]]
        graph[n1, n2] = c[2]
        graph[n2, n1] = c[2]
    distrs = np.stack(distrs)

    return nodes, start_loc, graph, objs, distrs

################################################################################





######################## Core Scavenger Hunt simulation ########################

class map():
    """
    Core map object that has method to initialize the object locations
    and maintain the distribution map. 
    """
    def __init__(self, nodes, start_loc, graph, objs, distrs):
        self.nodes = nodes
        self.N = len(nodes)
        self.graph = graph
        self.objs = objs
        self.K = len(objs)
        self.distrs = distrs
        self.start_loc = nodes[start_loc]
       
    def reset(self):
        self.cur_loc = self.start_loc
        self.obj_loc = [0]*self.K
        self.obj_list = [True]*self.K # objects to be found
        for o in range(self.K):
            p = random.random()
            for n in range(self.N):
                if p<sum(self.distrs[o,:n+1]):
                    self.obj_loc[o] = n
                    break
        
        self.cur_distrs = self.distrs.copy()
        self.observe()

        if sum(self.obj_list)==0:
            self.reset()

    def observe(self):
        # observe the object at current location
        # return c as the number of objects being found
        c = 0
        self.cur_distrs[:,self.cur_loc] = 0
        for o, n in enumerate(self.obj_loc):
            if n == self.cur_loc and self.obj_list[o]:
                c += 1
                self.cur_distrs[o,:] = 0
                self.obj_list[o] = False

        # update the distribution map
        for i, distr in enumerate(self.cur_distrs):
            s = sum(distr)
            if s>0:
                self.cur_distrs[i,:] = distr/s
        
        return c

    def get_object_list(self):
        # get the objects to be found
        return self.obj_list

    def get_object_loc(self):
        # get the true objects locations
        return self.obj_loc

    def move(self, n):
        # move to a node
        # return cost and the number of object being found
        cost = self.get_cost(self.cur_loc, n)
        self.cur_loc = n
        c = self.observe()

        return cost, c

    def get_cost(self, n1, n2):
        # get cost between two nodes
        return self.graph[n1, n2]

    def get_cost_map(self):
        # cost to all other nodes relative to the current node
        n = self.get_current_loc()
        return [self.graph[n1, n] for n1 in range(self.N)]

    def get_prob_distrs(self):
        # get distribution map represented as [# of object, # of node]
        # 2d numpy array
        return self.cur_distrs

    def get_find_at_least_one_prob(self):
        # get probability of finding at least one object at each of the nodes
        return 1-np.prod(1-self.cur_distrs, axis=0)

    def get_distrs(self):
        # get the prior distribution
        return self.distrs

    def get_node_num(self):
        return self.N

    def get_current_loc(self):
        return self.cur_loc

def load_map(fname):
    nodes, start_loc, graph, objs, distrs = parse_world(fname)
    return map(nodes, start_loc, graph, objs, distrs)

def reset_distribution(fname, nodes_range, cost_range,\
                          objects_range, occurrences_range):
    """
    reload a map with nodes config the same as the map given by fname
    but distribution resampled. 
    """
    generate("maps/distribution.dat", nodes_range, cost_range,\
             objects_range, occurrences_range)
    nodes, start_loc, graph, _, _ = parse_world(fname)
    _, _, _, objs, distrs = parse_world("maps/distribution.dat")

    return map(nodes, start_loc, graph, objs, distrs)

if __name__ == "__main__":
    nodes_range = [8, 8]
    cost_range = [10, 500]
    objects_range = [4, 4]
    occurrences_range = [1, 4]

    import sys
    generate(sys.argv[1],
        nodes_range, cost_range, objects_range, occurrences_range
    )
