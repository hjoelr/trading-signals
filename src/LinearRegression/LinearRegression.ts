import Big, {BigSource} from 'big.js';
import {NotEnoughDataError} from '../error';

export interface Point {
    readonly x: Big | number;
    readonly y: Big | number;
}

class CalculatedPoint implements Point {
    public readonly x: Big;
    public readonly y: Big;
    public readonly xy: Big;
    public readonly x_squared: Big;
    public readonly y_squared: Big;

    constructor(point: Point) {
        this.x = new Big(point.x);
        this.y = new Big(point.y);
        this.xy = new Big(point.x).mul(point.y);
        this.x_squared = new Big(point.x).pow(2);
        this.y_squared = new Big(point.y).pow(2);
    }
}

/**
 * Linear Regression attempts to model the relationship between two variables by fitting a linear equation
 * to observed data. A linear regression line has an equation of the form Y = bX + a, where X is the
 * explanatory variable and Y is the dependent variable. The slope of the line is b, and a is the intercept
 * (the value of Y when X = 0).
 */
export class LinearRegression {

    /**
     * Our list of points.
     */
    protected points: CalculatedPoint[] = [];

    /**
     * The a value in the linear regression formula: Y = bX + a  (where X is the value at x-axis).
     */
    protected a: Big | undefined;

    /**
     * The b value in the linear regression formula: Y = bX + a  (where X is the value at x-axis).
     */
    protected b: Big | undefined;

    /**
     * The stored standard deviation. Also called the Root Mean Square Deviation (RMSD).
     */
    protected residual: number | undefined;

    /**
     * The Pearson's R coefficient.
     */
    protected pearsons_r: number | undefined;

    /*
     * The following variables are used to minimize repeat calculations and improve performance.
     */

    /**
     * A summation of the x values of all the points.
     */
    protected sum_x: Big = new Big(0);

    /**
     * A summation of the y values of all the points.
     */
    protected sum_y: Big = new Big(0);

    /**
     * A summation of the x * y values of all the points.
     */
    protected sum_xy: Big = new Big(0);

    /**
     * A summation of the x² values of all the points.
     */
    protected sum_x_squared: Big = new Big(0);

    /**
     * A summation of the y² values of all the points.
     */
    protected sum_y_squared: Big = new Big(0);

    constructor(points?: Point | Point[]) {
        if (points) {
            this.push(points);
        }
    }

    /**
     * Add the sample data points to be evaluated.
     * 
     * @param points The sample points from which the linear regression is derived.
     */
    public push(points: Point | Point[]) {
        // Make sure we have array format
        if (!Array.isArray(points)) {
            points = [points];
        }

        let calculated_cleared: boolean = false;

        for (const point of points) {
            const calc_point = new CalculatedPoint(point);
            this.points.push(calc_point);

            this.sum_x = this.sum_x.add(calc_point.x);
            this.sum_y = this.sum_y.add(calc_point.y);
            this.sum_xy = this.sum_xy.add(calc_point.xy);
            this.sum_x_squared = this.sum_x_squared.add(calc_point.x_squared);
            this.sum_y_squared = this.sum_y_squared.add(calc_point.y_squared);

            // Only need to clear the calculated values once. Performance optimization.
            if (!calculated_cleared) {
                this.clearCalculated();
                calculated_cleared = true;
            }
        }
    }

    /**
     * Remove the first point off the list.
     */
    public shift() {
        const point = this.points.shift();

        if (point) {
            this.sum_x = this.sum_x.minus(point.x);
            this.sum_y = this.sum_y.minus(point.y);
            this.sum_xy = this.sum_xy.minus(point.xy);
            this.sum_x_squared = this.sum_x_squared.minus(point.x_squared);
            this.sum_y_squared = this.sum_y_squared.minus(point.y_squared);

            this.clearCalculated();
        }
    }

    /**
     * Performs the calculations to determine the the linear regression coefficients, the
     * standard deviation, and Pearson's R coefficient.
     */
    public calculate(): LinearRegression {
        // We can't calculate a line with only one point.
        if (this.points.length < 2) {
            throw new NotEnoughDataError();
        }

        // Since both the a and b equations have the same denominator, we calculate it once and use it in both.
        let denom: Big = new Big(this.points.length).mul(this.sum_x_squared).minus(this.sum_x.pow(2));

        // Source: https://www.statisticshowto.com/probability-and-statistics/regression-analysis/find-a-linear-regression-equation/
        //     (∑y) * (∑x²) - (∑x) * (∑xy)
        // a = ---------------------------
        //         n * (∑x²) - (∑x)²
        this.a = this.sum_y.mul(this.sum_x_squared).minus(this.sum_x.mul(this.sum_xy)).div(denom);

        // Source: same as above
        //     n * (∑xy) - (∑x) * (∑y)
        // b = ---------------------------
        //         n * (∑x²) - (∑x)²
        this.b = new Big(this.points.length).mul(this.sum_xy).minus(this.sum_x.mul(this.sum_y)).div(denom);

        // Calculate residual and Pearson's R
        let sum_residual_squared: Big = new Big(0);
        let mean_x: Big = this.sum_x.div(this.points.length);
        let mean_y: Big = this.sum_y.div(this.points.length);
        let sum_r_top: Big = new Big(0);
        let sum_x_minus_mean_x_squared: Big = new Big(0);
        let sum_y_minus_mean_y_squared: Big = new Big(0);

        for (const key in this.points) {
            if (Object.prototype.hasOwnProperty.call(this.points, key)) {
                const y_hat = this.getValue(this.points[key].x);

                sum_residual_squared = sum_residual_squared.add(this.points[key].y.minus(y_hat).pow(2));

                sum_r_top = sum_r_top.add(this.points[key].x.minus(mean_x).mul(this.points[key].y.minus(mean_y)));
                sum_x_minus_mean_x_squared = sum_x_minus_mean_x_squared.add(this.points[key].x.minus(mean_x).pow(2));
                sum_y_minus_mean_y_squared = sum_y_minus_mean_y_squared.add(this.points[key].y.minus(mean_y).pow(2));
            }
        }

        // Source: https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/v/standard-dev-residuals
        // residual = sqrt(
        //               ∑(y - (b * x + a))²     where (b * x + a) is the y value at x time along the linear regression line
        //               -------------------
        //                       n-2
        //            ) // sqrt
        this.residual = sum_residual_squared.div(this.points.length - 2).sqrt().toNumber();

        // Source: https://youtu.be/2SCg8Kuh0tE
        //                  ∑((x - x_mean) * (y - y_mean))
        // pearsons_r = -------------------------------------
        //              sqrt(∑(x - x_mean)² * ∑(y - y_mean)²)
        this.pearsons_r = sum_r_top.div(sum_x_minus_mean_x_squared.mul(sum_y_minus_mean_y_squared).sqrt()).toNumber();

        return this;
    }

    /**
     * Retrieve the standard deviation of the residuals. This is also called the
     * Root mean square deviation (RMSD).
     */
    public getResidual(std_dev_val: number = 1): number {
        if (!this.isCalculated()) {
            this.calculate();
        }

        return this.residual! * std_dev_val;
    }

    /**
     * Retrieve the standard deviation of the residuals. This is also called the
     * Root mean square deviation (RMSD).
     */
    public getStandardDeviation(std_dev_val: number = 1): number {
        return this.getResidual(std_dev_val);
    }

    /**
     * Retrieve the Pearson's R coefficient.
     * 
     * This value is helpful in determining if the data values are correlated to the
     * time (or x-axis) values.
     */
    public getPearsonsR(): number {
        if (!this.isCalculated()) {
            this.calculate();
        }

        return this.pearsons_r!;
    }

    /**
     * Return the value at the given time.
     * 
     * @param time The time value. Can be thought of as the x-axis value.
     */
    public getValue(time: number | Big): number {
        if (!(this.a && this.b)) {
            this.calculate();
        }

        // Y = bX + a
        return this.b!.mul(time).plus(this.a!).toNumber();
    }

    public getValues(time: number | Big, std_dev_val: number = 2) {
        return [this.getStandardDeviationValue(time, std_dev_val * -1), this.getValue(time), this.getStandardDeviationValue(time, std_dev_val)];
    }

    /**
     * Return the value that is so many standard deviations away from the middle.
     * 
     * @param time The time value. Can be thought of as the x-axis value.
     * @param std_deviation The number of standard deviations for which to return the value.
     *          A negative number will return the standard deviation value on the bottom side.
     */
    public getStandardDeviationValue(time: number | Big, std_deviation: number) {
        return this.getValue(time) + this.getStandardDeviation(std_deviation);
    }

    /**
     * The number of points represented.
     */
    public count(): number {
        return this.points.length;
    }

    /**
     * Resets all the calculated properties.
     */
    private clearCalculated() {
        this.a = this.b = this.residual = this.pearsons_r = undefined;
    }

    /**
     * Indicates if the points have been calculated.
     */
    private isCalculated() {
        return (this.a && this.b && this.residual && this.pearsons_r);
    }
}
