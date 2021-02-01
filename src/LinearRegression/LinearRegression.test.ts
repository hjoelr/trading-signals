// import Big from 'big.js';
import {LinearRegression, NotEnoughDataError} from '..';

import candles from '../test/fixtures/LinearRegression/candles.json';
import results from '../test/fixtures/LinearRegression/results.json';

describe('LinearRegression', () => {
  describe('calculate', () => {
    it('correctly calculates linear coefficients a and b', () => {
      const lr = new LinearRegression(candles.ascending1).calculate();
      expect(lr["a"]?.eq('-104.76288717'));
      expect(lr["b"]?.eq('0.00000006521944444444'));
    });

    it('returns the proper values along the linear regression line', () => {
      const lr = new LinearRegression(candles.ascending1);
      // Time minus 3
      expect(lr.getValue(1610856000) === 0.2962462299928406);
      // Time at first value
      expect(lr.getValue(1610866800) === 0.29695059999284057);
      // Time at last value
      expect(lr.getValue(1610953200) === 0.3025855599928402);
      // Time plus 3
      expect(lr.getValue(1610964000) === 0.30328992999284016);
    });

    it('correctly calculates the standard deviation (residual)', () => {
      const lr = new LinearRegression(candles.ascending1);
      expect(lr.getResidual() === 0.0045072988272163725);
    });

    it('correctly calculates Pearson\'s R coefficient', () => {
      const lr = new LinearRegression(candles.ascending1).calculate();
      expect(lr.getPearsonsR() === 0.3646586929169373);
    });

    it('can accurately handles pushed data', () => {
      const lr = new LinearRegression();
      lr.push(candles.ascending1);
      expect(lr.getResidual() === 0.0045072988272163725);
      expect(lr.getPearsonsR() === 0.3646586929169373);
    });

    it('fails with an error if not enough information exists when a calculation is attempted', () => {
      const lr = new LinearRegression({x:17, y:94});
      try {
        lr.calculate();
        fail('Expected error');
      } catch (error) {
        expect(error).toBeInstanceOf(NotEnoughDataError);
      }
    });
  });
});
