import unittest

import server


class FleetStatusTests(unittest.TestCase):
    def setUp(self):
        server.CFG = {
            "server": {
                "title": "Test Fleet",
                "subtitle": "",
                "browser_refresh_ms": 2500,
            },
            "nodes": [{"key": "node-1"}],
            "models": [],
        }
        server.STATE = {
            "nodes": {"node-1": self.node_state()},
            "models": {},
            "switch": None,
        }
        server.TOKEN_STORE = None
        server._hist.clear()

    @staticmethod
    def node_state(gpu_temp=60.0, host_temp=60.0, reachable=True):
        return {
            "key": "node-1",
            "name": "Test Node",
            "reachable": reachable,
            "gpus": [{"index": 0, "temp": gpu_temp, "power": 10.0}],
            "cpu_temp": host_temp,
            "temp_warn": 70,
            "temp_hot": 84,
        }

    def status(self):
        return server.snapshot()["agg"]

    def test_healthy_fleet_is_ok(self):
        agg = self.status()
        self.assertEqual("ok", agg["status"])
        self.assertTrue(agg["all_ok"])
        self.assertEqual([], agg["issues"])

    def test_warm_gpu_sets_warning(self):
        server.STATE["nodes"]["node-1"] = self.node_state(gpu_temp=70.0)
        agg = self.status()
        self.assertEqual("warning", agg["status"])
        self.assertFalse(agg["all_ok"])
        self.assertIn("GPU0 warm", agg["issues"][0])

    def test_hot_host_sets_hot(self):
        server.STATE["nodes"]["node-1"] = self.node_state(host_temp=90.0)
        agg = self.status()
        self.assertEqual("hot", agg["status"])
        self.assertIn("host hot", agg["issues"][0])

    def test_unreachable_node_is_degraded(self):
        server.STATE["nodes"]["node-1"] = self.node_state(reachable=False)
        agg = self.status()
        self.assertEqual("degraded", agg["status"])
        self.assertIn("unreachable", agg["issues"][0])

    def test_unavailable_model_is_degraded(self):
        server.CFG["models"] = [{
            "key": "model-1",
            "node": "node-1",
            "group": None,
        }]
        server.STATE["models"]["model-1"] = {
            "key": "model-1",
            "label": "Test Model",
            "reachable": False,
        }
        agg = self.status()
        self.assertEqual("degraded", agg["status"])
        self.assertIn("Test Model unavailable", agg["issues"])


if __name__ == "__main__":
    unittest.main()
