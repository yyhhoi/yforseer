from yforseer.evaluate import evaluate_stock_trend_prediction
import unittest
import numpy as np

class TestEvaluate(unittest.TestCase):
    def test_evaluate(self):
        xlast = np.array(
            [5, 10, 20, 200, 350, 100]
        )
        ypred = np.array(
            [20, 2, 24, 180, 450, 50]
        )
        ytest = np.array(
            [10, 7, 18, 210, 500, 10]
        )
        # over prediction up
        # over prediction down
        # wrong trend: up instead of down
        # wrong trend: down instead of up
        # under prediction up
        # under prediction down
        xlast_batch = np.repeat(xlast.reshape(1, -1) , axis=0, repeats=2)
        ypred_batch = np.repeat(ypred.reshape(1, -1) , axis=0, repeats=2)
        ytest_batch = np.repeat(ytest.reshape(1, -1) , axis=0, repeats=2)
        a, b, c = evaluate_stock_trend_prediction(xlast, ypred, ytest, batch=False)
        d, e, f = evaluate_stock_trend_prediction(xlast_batch, ypred_batch, ytest_batch, batch=True)

        np.testing.assert_array_almost_equal(a, 0.6666666666666666)
        np.testing.assert_array_almost_equal(b, 0.8900468384074941)
        np.testing.assert_array_almost_equal(c, 0.4892857142857143)
        np.testing.assert_array_almost_equal(d, np.array([0.66666667, 0.66666667]))
        np.testing.assert_array_almost_equal(e, np.array([0.89004684, 0.89004684]))
        np.testing.assert_array_almost_equal(f, np.array([0.48928571, 0.48928571]))



if __name__ == '__main__':
    unittest.main()