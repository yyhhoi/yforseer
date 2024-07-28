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

        # Test non batch mode
        expected_buy_returns = np.array([1., 0.89004684, 0.89004684])
        expected_perfect_buy_returns = np.array([0.99999999, 0.76782376, 0.76782376])
        expected_sell_returns = np.array([0.3, 0.48928571, 0.48928571])
        expected_perfect_sell_returns = np.array([0.29999999, 0.71799999, 0.71799999])
        np.testing.assert_array_almost_equal(a[0], 0.6666666666666666)
        np.testing.assert_array_almost_equal(a[1], 0.5)
        np.testing.assert_array_almost_equal(a[2], 0.5)
        np.testing.assert_array_almost_equal(b[0], expected_buy_returns)
        np.testing.assert_array_almost_equal(b[1], expected_perfect_buy_returns)
        np.testing.assert_array_almost_equal(c[0], expected_sell_returns)
        np.testing.assert_array_almost_equal(c[1], expected_perfect_sell_returns)

        # Test batch mode
        d, e, f = evaluate_stock_trend_prediction(xlast_batch, ypred_batch, ytest_batch, batch=True)
        expected_buy_returns = np.array([[1., 0.89004684, 0.89004684], [1., 0.89004684, 0.89004684]])
        expected_perfect_buy_returns = np.array([[0.99999999, 0.76782376, 0.76782376], [0.99999999, 0.76782376, 0.76782376]])
        expected_sell_returns = np.array([[0.3, 0.48928571 , 0.48928571], [0.3, 0.48928571 , 0.48928571]])
        expected_perfect_sell_returns = np.array([[0.29999999 , 0.71799999 , 0.71799999], [0.29999999 , 0.71799999 , 0.71799999]])
        np.testing.assert_array_almost_equal(d[0], np.array([0.66666667, 0.66666667]))
        np.testing.assert_array_almost_equal(d[1], np.array([0.5, 0.5]))
        np.testing.assert_array_almost_equal(d[2], np.array([0.5, 0.5]))
        np.testing.assert_array_almost_equal(e[0], expected_buy_returns)
        np.testing.assert_array_almost_equal(e[1], expected_perfect_buy_returns)
        np.testing.assert_array_almost_equal(f[0], expected_sell_returns)
        np.testing.assert_array_almost_equal(f[1], expected_perfect_sell_returns)



if __name__ == '__main__':
    unittest.main()