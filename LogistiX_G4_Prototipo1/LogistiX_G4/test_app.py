import unittest
from app import predict_order, SCHEMA, PIPELINE
import pandas as pd

class TestPrediction(unittest.TestCase):
    def test_probabilidad_y_umbral(self):
        p, flag = predict_order(23,94,117,'Alta')
        self.assertTrue(0 <= p <= 1)
        self.assertEqual(flag,p >= SCHEMA['umbral'])
    def test_mismo_pipeline(self):
        row=pd.DataFrame([[37,71,219,'Urgente']],columns=SCHEMA['variables'])
        p,_=predict_order(37,71,219,'Urgente')
        self.assertAlmostEqual(p,float(PIPELINE.predict_proba(row)[0,1]))
    def test_rechaza_invalido(self):
        for values in [(0,94,117,'Alta'),(23,101,117,'Alta'),(23,94,251,'Alta'),(23,94,117,'Otra'),(23.5,94,117,'Alta')]:
            with self.assertRaises(ValueError): predict_order(*values)

if __name__=='__main__': unittest.main()
