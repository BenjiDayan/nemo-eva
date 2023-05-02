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
import pandas as pd

import sys

from girg_sampling import girgs

sys.path.append('/cluster/home/bdayan/girgs/')
from benji_src.benji_girgs import fitting, generation, utils

import os
import time

from abstract_stage import AbstractStage
from graph_crawler import GraphCrawler
from helpers.print_blocker import PrintBlocker



PrintBlocker = utils.HiddenPrints


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
        try:
            self.results_df = pd.read_csv(self.resultspath)
        except Exception:
            print('no results_df')
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
    # This function is wacky??
    def binary_search(self, goal_f, goal, a, b, f_a=None, f_b=None, depth=0):
        if f_a is None:
            f_a = goal_f(a)
        if f_b is None:
            f_b = goal_f(b)
        m = (a + b) / 2
        f_m = goal_f(m)
        # Check for monotonicity
        if depth < 10 and (f_a <= f_m <= f_b or f_a >= f_m >= f_b):
            # whichever we're sandwiched is the better one
            if f_a <= goal <= f_m or f_a >= goal >= f_m:
                 out, hist_out = self.binary_search(
                    goal_f, goal,
                    a, m, f_a, f_m,
                    depth=depth+1)
                 return out, [(m, f_m)] + hist_out
            # no sandwich, somehow we assume that goal > f_a, f_b, f_m
            else:
                out, hist_out = self.binary_search(
                    goal_f, goal,
                    m, b, f_m, f_b,
                    depth=depth+1)
                return out, [(m, f_m)] + hist_out
        # Non monotonicity means things are quite close. Somehow here we assume that
        # goal < f_a, f_b, f_m
        out = min([(a, f_a), (b, f_b), (m, f_m)], key=lambda x: x[1])
        return out, [out]


    def binary_search_better(self, goal_f, goal, a, b, f_a=None, f_b=None, depth=0, verbose=False):
        """This search makes no assumptions on monotonicity, is just a kinda basic idea. But
        Still seems better than the previous one :P"""
        if f_a is None:
            f_a = goal_f(a, depth=depth)
        if f_b is None:
            f_b = goal_f(b, depth=depth)
        m = (a + b) / 2
        f_m = goal_f(m, depth=depth)

        if verbose:
            print(f'a: {a}, m: {m}, b: {b}, f_a: {f_a}, f_m: {f_m}, f_b: {f_b}, goal: {goal}')

        if depth < 10:
            # if goal > max(f_a, f_m, f_b) or goal_f < min(f_a, f_m, f_b):
            # Either:
            #   1. goal is outside all - go for the side which is closer to goal
            #   2. goal is somewhere between peeps.
            #       if e.g. _ - ` then we should check which side enclosure
            #       if e.g. - ' _  then could be both enclosure or just one.
            #
            a_closer = abs(goal - f_a) < abs(goal - f_b)
            a_m_enclosure = f_a <= goal <= f_m or f_a >= goal >= f_m
            m_b_enclosure = f_m <= goal <= f_b or f_m >= goal >= f_b
            if a_m_enclosure or m_b_enclosure:  # Case 2: enclosed by one or both sides
                a_side = a_m_enclosure  # If both sides enclosed then either is good.
            else:  # Case 1: not enclosed by either side so just go to the closest one
                a_side = a_closer
            lo, f_lo = (a, f_a) if a_side else (m, f_m)
            hi, f_hi = (m, f_m) if a_side else (b, f_b)
            out, hist_out = self.binary_search_better(
                goal_f, goal,
                lo, hi, f_lo, f_hi,
                depth=depth+1, verbose=verbose)

            return out, [(m, f_m)] + hist_out

        # max depth exceeded, return best guess - closest to goal. This is easy if goal is
        # goal_f < f_* or goal_f > f_*
        out = min([(a, f_a), (b, f_b), (m, f_m)], key=lambda x: abs(goal - x[1]))
        return out, [out]

    def half_double_search(self, goal_f, goal, a, b, m, f_a=None, f_b=None, f_m=None, depth=0, verbose=False):
        """We assume that goal_f is monotonic in general as a function of input. E.g. it's an increasing
        function if input is const and criterium is avg degree

        We start off with a < m < b. We half/double search until the goal is somewhere within f_a, f_m, f_b,
        i.e. we have enclosed. I.e. we pick the side which is closer to the goal,
        e.g. f_a closer, then do a <- a/2, m <- a, b <- 2a (probably == m).
        or if f_b closer, do a <- b/2 (probably == m), m <- b, b <- 2b.
        Once we have overshot, we do a binary search.
        """
        if f_a is None:
            f_a = goal_f(a)
        if f_b is None:
            f_b = goal_f(b)
        if f_m is None:
            f_m = goal_f(m)

        if verbose:
            print(f'a: {a}, m: {m}, b: {b}, f_a: {f_a}, f_m: {f_m}, f_b: {f_b}, goal: {goal}')

        if depth < 10:
            a_closer = abs(goal - f_a) < abs(goal - f_b)
            a_m_enclosure = f_a <= goal <= f_m or f_a >= goal >= f_m
            m_b_enclosure = f_m <= goal <= f_b or f_m >= goal >= f_b
            enclosed = a_m_enclosure or m_b_enclosure
            if not enclosed:
                if a_closer:
                    a2, f_a2 = (a/2, None)
                    m2, f_m2 = (a, f_a)
                    b2, f_b2 = (2*a, f_m if m == 2*a else None)
                else:
                    a2, f_a2 = (b/2, f_m if m == b/2 else None)
                    m2, f_m2 = (b, f_b)
                    b2, f_b2 = (2*b, None)
                out, hist_out = self.half_double_search(
                    goal_f, goal,
                    a2, b2, m2, f_a2, f_b2, f_m2,
                    depth=depth+1, verbose=verbose)
            else:  # enclosed. Could be enclosed in both sides in which case just default to a_m_enclosure
                lo, f_lo = (a, f_a) if a_m_enclosure else (m, f_m)
                hi, f_hi = (m, f_m) if a_m_enclosure else (b, f_b)
                out, hist_out = self.binary_search_better(
                    goal_f, goal,
                    lo, hi, f_lo, f_hi,
                    depth=depth+1, verbose=verbose)

            return out, [(m, f_m)] + hist_out

        out = min([(a, f_a), (b, f_b), (m, f_m)], key=lambda x: abs(goal - x[1]))
        return out

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
                # return networkit.globals.clustering(h)
                return utils.LCC(h)
        goal = criterium(g)

        def guess_goal(t, *args, **kwargs):
            hyper_t = networkit.generators.HyperbolicGenerator(
                n_hyper, k, gamma, t).generate()
            hyper_t = self.shrink_to_giant_component(hyper_t)
            return criterium(hyper_t)
        (t, crit_diff), hist = self.binary_search(guess_goal, goal, 0, 0.99)
        hist = [(1 / t, crit_diff) for t, crit_diff in hist]
        hyper = networkit.generators.HyperbolicGenerator(
            n_hyper, k, gamma, t).generate()
        info_map = [
            ("n", n_hyper),
            ("k", k),
            ("gamma", gamma),
            ("t", t),
            ("target_lcc", goal),
            ("fit_lcc", criterium(hyper)),
            ("hist", hist)
        ]
        info = "|".join([name + "=" + str(val) for name, val in info_map])
        return (info, hyper)

    def fit_ndgirg_cube_const(self, d, alpha, target_avg_degree, verbose=False):
        """Returns a func that will find the right const for target degree and alpha"""
        def fit_girg(weights, const=None, outer_depth=None):
            """If given const, we suppose that this is a good starting point to search from, otherwise we start from 1.0
            Starting from given const we take fewer steps so starting depth is > 0: we're trying to use
            2 + outer_depth//2, it might be ok.
            """
            goal = target_avg_degree
            n, tau = len(weights), None
            criterium = lambda g: utils.avg_degree(g)
            def guess_goal(const, *args, **kwargs):
                g_out, const = generation.cgirg_gen_cube(n, d, tau, alpha, const=const, weights=weights)
                return criterium(g_out)

            # FIXME do we want to put this hack back in?
            # depth = 0 if const is None else 2 + outer_depth//2
            depth = 5
            if verbose:
                print(f'starting search at const={const}; depth={depth}')

            if const is None:
                (const, crit_diff), hist = self.half_double_search(guess_goal, goal, 0.5, 2.0, 1.0, depth=depth, verbose=verbose)
            else:
                (const, crit_diff), hist = self.half_double_search(guess_goal, goal, 0.5*const, 2*const, const, depth=depth,verbose=verbose)

            g_out, const = generation.cgirg_gen_cube(n, d, tau, alpha, const=const, weights=weights)
            return g_out, const, crit_diff, hist
        return fit_girg

    def fit_ndgirg_general(self, d, criterium, cube=False, verbose=False):
        """

        Args:
            d: dimension of GIRG to be fit
            criterium: e.g. utils.LCC
            cube: if True then we use the cube version of GIRG - must fit const as well using self.fit_ndgirg_cube_const
                - otherwise const is simply fitted by girgs.scaleWeights
            verbose: For printing

        Returns: A function that, given a real graph g, will fit alpha (and const

        """
        def fit_girg(g, *args, **kwargs):
            start_time = time.time()
            n = g.numberOfNodes()
            target_avg_degree = utils.avg_degree(g)
            tau = utils.powerlaw_fit_graph(g)
            # TODO is this a good idea? Or even cap also at 3.0?
            tau = max(tau, 2.1)

            goal = criterium(g)
            const = None

            def guess_goal(t, depth=None):
                # FIXME do we want to put this hack back in?
                nonlocal const  # updated guess for cube const fitting. ignored for non cube GIRG

                alpha = 1 / t
                if cube == False:
                    g_out, _, _, _, _, _ = generation.cgirg_gen(n, d, tau, alpha, desiredAvgDegree=target_avg_degree)
                else:
                    # we need 2**d sized weights, e.g. d=2 then the [0, 0.5] x [0, 0.5] has 1/4 of the total points
                    weights = girgs.generateWeights((2**d)*n, tau)
                    # FIXME do we want to put this hack back in?
                    # const_guess = const if not t in [0.1, 0.99] else None

                    # * 1.3 rough guess because cgirg_gen_cube will have lower degrees than cgirg_gen
                    const_guess = girgs.scaleWeights(weights, target_avg_degree, d, alpha) * 1.3
                    g_out, const, crit_diff, const_hist = \
                        self.fit_ndgirg_cube_const(
                            d, alpha, target_avg_degree, verbose=verbose
                        )(weights, const=const_guess, outer_depth=depth)

                    # FIXME mild hack to make nonlocal const only track for the alpha = m middle of binsearch
                    # if not t in [0.1, 0.99]:  # don't update the nonlocal const - i.e. have to revert it back
                    #     const = const_new

                return criterium(g_out)

            (t, crit_diff), hist = self.binary_search_better(guess_goal, goal, 0.01, 0.99, verbose=verbose)
            alpha = 1/t
            hist = [(1/t, crit_diff) for t, crit_diff in hist]

            if cube == False:
                g_out, _, _, _, const, _ = generation.cgirg_gen(n, d, tau, alpha, desiredAvgDegree=target_avg_degree)
            else:
                weights = girgs.generateWeights((2**d)*n, tau)
                g_out, _ = generation.cgirg_gen_cube((2**d)*n, d, tau, alpha, const=const, weights=weights)

            info_map = [
                ("tau", tau),
                ("alpha", alpha),
                ("const", const),
                ("target_lcc", goal),
                ("fit_lcc", criterium(g_out)),
                ("fitting_time", time.time() - start_time),
                ("hist", hist),
            ]
            info = "|".join([name + "=" + str(val) for name, val in info_map])
            return (info, g_out)
        return fit_girg
    def fit_ndgirg_binsearch(self, d):
        def criterium(g):
            with PrintBlocker():
                # return networkit.globals.clustering(g)
                return utils.LCC(g)
        fit_girg = self.fit_ndgirg_general(d, criterium)
        return fit_girg


    def fit_ndgirg(self, d):
        def fit_girg(g, *args, **kwargs):
            n = g.numberOfNodes()
            alpha, const, tau, hist, target_lcc, fit = fitting.fit_cgirg(g, d, *args, **kwargs)
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
        ]

        for d in range(1, 4):
            model_types.append(
                (f"{d}d-girg", 
                 self.fit_ndgirg_binsearch(d)
                )
            )

        for d in range(1, 3):
            model_types.append(
                (f"{d}d-cube-girg",
                 self.fit_ndgirg_general(d, utils.LCC, cube=True)
                )
            )

        outputs = []
        # all_keys = set()
        for model_name, model_converter in model_types:
            if self.results_df.loc[
                    (self.results_df["Graph"] == graph_dict["Name"]) &
                    (self.results_df["Model"] == model_name)].shape[0] > 0:
                print("Skipping", model_name, "for", graph_dict["Name"])
                continue
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

        pool = multiprocessing.Pool(10)
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
        with open(self.resultspath, "a") as results_file:
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