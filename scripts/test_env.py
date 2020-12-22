import sys
from os.path import join, abspath, dirname
sys.path.append(dirname(dirname(join(abspath(__file__)))))
import map
from map import load_map, parse_world

if __name__ == "__main__":
    m = load_map("maps/default.dat")
    m.reset()
    print(m.get_cost(0,1))
    print(m.get_cost_map())
    print(m.get_prob_distrs())
    print(m.get_distrs())
    print(m.get_node_num())
    print(m.get_current_loc())
    print(m.get_find_at_least_one_prob())
    print(m.obj_loc)

    print("\n")
    print(m.move(1))
    print(m.get_cost(0,1))
    print(m.get_cost_map())
    print(m.get_prob_distrs())
    print(m.get_distrs())
    print(m.get_node_num())
    print(m.get_current_loc())
    print(m.get_find_at_least_one_prob())


    print("\n")
    print(m.move(2))
    print(m.get_cost(0,1))
    print(m.get_cost_map())
    print(m.get_prob_distrs())
    print(m.get_distrs())
    print(m.get_node_num())
    print(m.get_current_loc())
    print(m.get_find_at_least_one_prob())
