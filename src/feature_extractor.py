import collections
import csv
import itertools
import math
import multiprocessing
import sys
import numpy
import powerlaw
import random
import networkit

import sys
sys.path.append('/cluster/home/bdayan/girgs/')
from benji_src.benji_girgs import fitting, generation

import os

from abstract_stage import AbstractStage
from graph_crawler import GraphCrawler
from helpers.print_blocker import PrintBlocker

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

PrintBlocker = HiddenPrints


class NoDaemonProcessPool(multiprocessing.pool.Pool):
    class NoDaemonProcess(multiprocessing.Process):
        def _get_daemon(self):
            return False

        def _set_daemon(self, value):
            pass
        daemon = property(_get_daemon, _set_daemon)

    Process = NoDaemonProcess


class FeatureExtractor(AbstractStage):
    _stage = "2-features"

    def __init__(self, graph_dicts):
        super(FeatureExtractor, self).__init__()
        self.graph_dicts = graph_dicts
        networkit.engineering.setNumberOfThreads(1)

    def getDeepValue(self, a_dict, keylist):
        for subkey in keylist:
            a_dict = a_dict[subkey]
        return a_dict

    # def shrink_to_giant_component(self, g):
    #     comp = networkit.components.ConnectedComponents(g)
    #     comp.run()
    #     giant_id = max(comp.getPartition().subsetSizeMap().items(),
    #                    key=lambda x: x[1])
    #     giant_comp = comp.getPartition().getMembers(giant_id[0])
    #     for v in g.nodes():
    #         if v not in giant_comp:
    #             for u in g.neighbors(v):
    #                 g.removeEdge(v, u)
    #             g.removeNode(v)
    #     name = g.getName()
    #     g = networkit.graph.GraphTools.getCompactedGraph(
    #       g,
    #       networkit.graph.GraphTools.getContinuousNodeIds(g)
    #     )
    #     g.setName(name)
    #     return g

    def shrink_to_giant_component(self, g):
        cc = networkit.components.ConnectedComponents(g)
        cc.run()
        return cc.extractLargestConnectedComponent(g, True)

    def analyze(self, g):
        # networkit.engineering.setNumberOfThreads(1)
        originally_weighted = g.isWeighted()
        if originally_weighted:
            g = g.toUnweighted()
        g.removeSelfLoops()
        g = self.shrink_to_giant_component(g)
        degrees = networkit.centrality.DegreeCentrality(g).run().scores()
        with PrintBlocker():
            fit = powerlaw.Fit(degrees, fit_method='Likelihood')
        stats = {
            "Originally Weighted": originally_weighted,
            "Degree Distribution": {
                "Powerlaw": {
                    "Alpha": fit.alpha,
                    "KS Distance": fit.power_law.KS()
                }
            }
        }

        #############

        # possible profiles: minimal, default, complete
        networkit.profiling.Profile.setParallel(1)
        networkit.setSeed(seed=42, useThreadId=False)
        pf = networkit.profiling.Profile.create(g, preset="complete")

        for statname in pf._Profile__measures.keys():
            stats[statname] = pf.getStat(statname)

        stats.update(pf._Profile__properties)

        keys = [[key] for key in stats.keys()]
        output = dict()
        while keys:
            key = keys.pop()
            val = self.getDeepValue(stats, key)
            if isinstance(val, dict):
                keys += [key + [subkey] for subkey in val]
            elif isinstance(val, int) or isinstance(val, float):
                output[".".join(key)] = val
            elif key == ['Diameter Range']:
                output['Diameter Min'] = val[0]
                output['Diameter Max'] = val[1]

        return output

    def binary_search(self, goal_f, goal, a, b, f_a=None, f_b=None, depth=0):
        if f_a is None:
            f_a = goal_f(a)
        if f_b is None:
            f_b = goal_f(b)
        m = (a + b) / 2
        f_m = goal_f(m)
        if depth < 10 and (f_a <= f_m <= f_b or f_a >= f_m >= f_b):
            if f_a <= goal <= f_m or f_a >= goal >= f_m:
                return self.binary_search(
                    goal_f, goal,
                    a, m, f_a, f_m,
                    depth=depth+1)
            else:
                return self.binary_search(
                    goal_f, goal,
                    m, b, f_m, f_b,
                    depth=depth+1)
        return min([(a, f_a), (b, f_b), (m, f_m)], key=lambda x: x[1])

    def fit_er(self, g):
        networkit.setSeed(seed=42, useThreadId=False)
        return networkit.generators.ErdosRenyiGenerator.fit(g).generate()

    def fit_ba(self, g, fully_connected_start):
        random.seed(42, version=2)
        networkit.setSeed(seed=42, useThreadId=False)
        # n, m = g.size()
        n, m = g.numberOfNodes(), g.numberOfEdges()
        m_0 = math.ceil(m / n)
        ba = networkit.Graph(n)
        nodes = list(ba.iterNodes())
        edges_added = 0
        if fully_connected_start:
            start_connections = itertools.combinations(nodes[:m_0], 2)
        else:  # circle
            start_connections = (
                [(nodes[m_0-1], nodes[0])] +
                [(nodes[i], nodes[i+1]) for i in range(m_0-1)]
            )
        for u, v in start_connections:
            ba.addEdge(u, v)
            edges_added += 1

        for i, v in list(enumerate(nodes))[m_0:]:
            num_new_edges = int((m-edges_added)/(n-i))
            really_new_edges = 0
            to_connect = set()
            while len(to_connect) < num_new_edges:
                num_draws = num_new_edges - len(to_connect)
                to_connect_draws = [
                    # random.choice(ba.randomEdge())
                    random.choice(networkit.graphtools.randomEdge(ba, True))
                    for i in range(num_draws)
                ]
                to_connect |= set(
                    u for u in to_connect_draws if not ba.hasEdge(v, u)
                )
            for u in to_connect:
                ba.addEdge(u, v)
            edges_added += num_new_edges
        return ba

    def fit_chung_lu(self, g):
        networkit.setSeed(seed=42, useThreadId=False)
        return networkit.generators.ChungLuGenerator.fit(g).generate()

    def fit_hyperbolic(self, g):
        networkit.setSeed(seed=42, useThreadId=False)
        degrees = networkit.centrality.DegreeCentrality(g).run().scores()
        with PrintBlocker():
            fit = powerlaw.Fit(degrees, fit_method='Likelihood')
        gamma = max(fit.alpha, 2.1)
        # n, m = g.size()
        n, m = g.numberOfNodes(), g.numberOfEdges()
        degree_counts = collections.Counter(degrees)
        n_hyper = n + max(0, 2*degree_counts[1] - degree_counts[2])
        k = 2 * m / (n_hyper-1)

        def criterium(h):
            with PrintBlocker():
                return networkit.globals.clustering(h)
        goal = criterium(g)

        def guess_goal(t):
            hyper_t = networkit.generators.HyperbolicGenerator(
                n_hyper, k, gamma, t).generate()
            hyper_t = self.shrink_to_giant_component(hyper_t)
            return criterium(hyper_t)
        t, crit_diff = self.binary_search(guess_goal, goal, 0, 0.99)
        hyper = networkit.generators.HyperbolicGenerator(
            n_hyper, k, gamma, t).generate()
        info_map = [
            ("n", n_hyper),
            ("k", k),
            ("gamma", gamma),
            ("t", t)
        ]
        info = "|".join([name + "=" + str(val) for name, val in info_map])
        return (info, hyper)
    

    def fit_ndgirg(self, d):
        def fit_girg(g):
            n = g.numberOfNodes()
            alpha, const, tau, hist, target_lcc, fit = fitting.fit_cgirg(g, d)
            g_out, _, _, _, _, _ = generation.cgirg_gen(n, d, tau, alpha, const=const)
            info_map = [
                ("tau", tau),
                ("alpha", alpha),
                ("const", const),
                ("fit", fit),
                ("target_lcc", target_lcc),
                ("hist", hist),
            ]
            info = "|".join([name + "=" + str(val) for name, val in info_map])
            return (info, g_out)
        return fit_girg

    
    # def fit_2dgirg(self, g):
    #     d = 2
    #     n = g.numberOfNodes()
    #     alpha, const, tau, hist, target_lcc, fit = fitting.fit_cgirg(g, d, max_fit_steps=15)
    #     g_out, _, _, _, _, _ = generation.cgirg_gen(n, d, tau, alpha, const=const)
    #     info_map = [
    #         ("tau", tau),
    #         ("alpha", alpha),
    #         ("const", const),
    #         ("fit", fit),
    #         ("target_lcc", target_lcc),
    #         ("hist", hist),
    #     ]
    #     info = "|".join([name + "=" + str(val) for name, val in info_map])
    #     return (info, g_out)

    def _execute_one_graph(self, graph_dict):
        # in_path = (
        #     GraphCrawler()._stagepath +
        #     graph_dict["Group"] + "/" +
        #     graph_dict["Path"])
        in_path = graph_dict["FullPath"]
        out_path = self._stagepath + "results.csv"
        graph_type = graph_dict["Group"]

        g = None
        # try:
        #     g = networkit.readGraph(
        #         in_path,
        #         networkit.Format.EdgeList,
        #         separator=" ",
        #         firstNode=0,
        #         commentPrefix="%",
        #         continuous=True)
        try:
            g = networkit.readGraph(in_path, networkit.Format.EdgeListSpaceOne)
        except Exception as e:
            print(e)

        if not g:
            print("could not import graph from path", in_path)
        if g.numberOfNodes() > 0 and g.numberOfEdges() > 0:
            if g.degree(0) == 0:
                g.removeNode(0)

        # print("Graph", g.toString())
        g = self.shrink_to_giant_component(g)
        if g.numberOfNodes() < 100:
            print(
                "Graph is too small (" +
                str(g.numberOfNodes()) +
                " nodes, needs 100): " +
                in_path)

        model_types = [
            ("real-world",
                lambda x: ("", x)),
            ("ER",
                lambda x: ("", self.fit_er(x))),
            ("BA circle",
                lambda x: ("", self.fit_ba(x, fully_connected_start=False))),
            ("BA full",
                lambda x: ("", self.fit_ba(x, fully_connected_start=True))),
            ("chung-lu",
                lambda x: ("", self.fit_chung_lu(x))),
            ("hyperbolic",
                self.fit_hyperbolic),
            # ("1d-girg",
            #     self.fit_1dgirg),
            # ("2d-girg",
            #     self.fit_2dgirg)    
        ]
        for d in range(1, 6):
            model_types.append(
                (f"{d}d-girg", 
                 lambda x: self.fit_ndgirg(d)(x)
                )
            )

        outputs = []
        # all_keys = set()
        for model_name, model_converter in model_types:
            try:
                info, model = model_converter(g)
                output = self.analyze(model)
            except ZeroDivisionError as e:
                print("Error:", e, "for", model_name, "of", graph_dict["Name"], model_name)
            except Exception as e:
                print("Error:", e, "for", model_name, "of", graph_dict["Name"], model_name)
            else:
                # output["Graph"] = g.getName()
                output["Graph"] = graph_dict["Name"]
                output["Type"] = graph_type
                output["Model"] = model_name
                output["Info"] = info
                self._save_as_csv(output)

        #     outputs.append(output)
        # return outputs
                # outputs.append((model_name, info, output))
                # all_keys |= set(output.keys())

        # for model_name, info, output in sorted(outputs):
            # for key in all_keys - set(output.keys()):
            #    output[key] = float("nan")

    def _execute(self):
        pool = multiprocessing.Pool(10)
        pool.map(self._execute_one_graph, self.graph_dicts)
        pool.close()
        pool.join()

    def execute_immediate_write(self):
        if not os.path.exists(self._stagepath):
            os.makedirs(self._stagepath)

        writer_pool = multiprocessing.Pool(1)
        writer_out = writer_pool.apply_async(self.listener, (self._dict_queue,))

        pool = multiprocessing.Pool(12)
        # pool.map(self._execute_one_graph, self.graph_dicts)

        jobs = []
        for graph_dict in self.graph_dicts:
            job = pool.apply_async(self._execute_one_graph, (graph_dict,))
            jobs.append(job)

        for job in jobs:
            job.get()
        
        self._dict_queue.put(None)

        pool.close()
        pool.join()
        writer_out.get()

        writer_pool.close()
        writer_pool.join()


    def listener(self, q):
        with open(self.resultspath, "w") as results_file:
            wrote_header = False
            while True:
                result_dict = q.get()
                if result_dict is None:
                    print('Done writing to csv')
                    break
                else:
                    print(f'Saving csv of result {result_dict["Graph"]}, {result_dict["Model"]}')

                all_keys = set(result_dict.keys())
                fieldnames = sorted(all_keys)

                dict_writer = csv.DictWriter(results_file, fieldnames)
                if not wrote_header:
                    dict_writer.writeheader()
                    wrote_header = True

                dict_writer.writerow(result_dict)
                results_file.flush()