import unittest

from static_agent_training.learning_trace import build_traces


class LearningTraceTests(unittest.TestCase):
    def test_controlled_trace_is_reproducible_and_records_distinct_learning_rules(self):
        old,new=build_traces()
        self.assertEqual((old,new),build_traces())
        expected=['observation','expectation','outcome','empirical_update','structural_correction','next_prediction']
        for trace in (old,new):
            self.assertEqual([s['stage'] for s in trace['stages']],expected)
            for stage in trace['stages'][:3]+trace['stages'][5:]:
                self.assertEqual(stage['before']['parameters'],stage['after']['parameters'])
            empirical=trace['stages'][3]
            self.assertFalse(empirical['before']['revised'])
            self.assertTrue(empirical['after']['revised'])
            self.assertEqual([e for e in empirical['events'] if e['kind']=='EMPIRICAL_UPDATE'][0]['correct'],False)
            structural=trace['stages'][4]
            self.assertEqual(structural['events'][0]['kind'],'CONFLICT')
            self.assertEqual(empirical['after']['modal_embedding'],structural['before']['modal_embedding'])
        self.assertEqual(old['stages'][3]['before']['parameters']['embedding'],old['stages'][3]['after']['parameters']['embedding'])
        for group in ('bias','weights','gates'):
            self.assertEqual(new['stages'][3]['before']['parameters'][group],new['stages'][3]['after']['parameters'][group])
        for trace in (old,new):
            source=trace['pair_aliases']['source']
            before=trace['stages'][3]['before']['parameters'];after=trace['stages'][3]['after']['parameters']
            group='output' if trace is old else 'prediction'
            self.assertNotEqual(before[group][source],after[group][source])
            for q in before[group]:
                if q!=source:self.assertEqual(before[group][q],after[group][q])
        self.assertNotEqual(old['stages'][3]['before']['prediction_mass'],old['stages'][3]['after']['prediction_mass'])
        self.assertEqual(new['stages'][3]['before']['prediction_mass'],new['stages'][3]['after']['prediction_mass'])
        self.assertEqual(old['stages'][3]['after']['modal_signal'],'obs_+5')
        self.assertEqual(new['stages'][3]['after']['modal_signal'],new['stages'][2]['detail']['observed'])
        # Equal correction to the checked logit, different spillover to other inputs.
        a,b=old['stages'][4],new['stages'][4]
        self.assertEqual(a['after']['modal_embedding'],b['after']['modal_embedding'])
        self.assertEqual(a['before']['input_embedding']['obs_+6'],a['after']['input_embedding']['obs_+6'])
        self.assertNotEqual(b['before']['input_embedding']['6'],b['after']['input_embedding']['6'])
        self.assertEqual(old['stages'][-1]['detail']['emitted'],'expect_+5')
        self.assertEqual(new['stages'][-1]['detail']['emitted'],new['stages'][2]['detail']['observed'])
