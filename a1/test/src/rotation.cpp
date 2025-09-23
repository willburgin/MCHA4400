#include <doctest/doctest.h>
#include <cmath>
#include <Eigen/Core>

#ifndef CAPTURE_EIGEN
#define CAPTURE_EIGEN(x) INFO(#x " = \n", x.format(Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]")));
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#include "../../src/rotation.hpp"

SCENARIO("rotx")
{
    GIVEN("x = 0")
    {
        Eigen::Matrix3d R;
        double x = 0;

        WHEN("Calling R = rotx(x)")
        {
            R = rotx(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                  -0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }
        }

        WHEN("Calling R = rotx(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = rotx(x, dRdx);

            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                  -0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(                   0));
                        CHECK(dRdx(1,0) == doctest::Approx(                   0));
                        CHECK(dRdx(2,0) == doctest::Approx(                   0));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                   0));
                        CHECK(dRdx(1,1) == doctest::Approx(                  -0));
                        CHECK(dRdx(2,1) == doctest::Approx(                   1));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   0));
                        CHECK(dRdx(1,2) == doctest::Approx(                  -1));
                        CHECK(dRdx(2,2) == doctest::Approx(                  -0));
                    }
                }
            }        
        }
    }

    GIVEN("x = 0.1")
    {
        Eigen::Matrix3d R;
        double x = 0.1;

        WHEN("Calling R = rotx(x)")
        {
            R = rotx(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(      0.995004165278));
                        CHECK(R(2,1) == doctest::Approx(     0.0998334166468));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(    -0.0998334166468));
                        CHECK(R(2,2) == doctest::Approx(      0.995004165278));
                    }
                }
            }
        }

        WHEN("Calling R = rotx(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = rotx(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);
                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(      0.995004165278));
                        CHECK(R(2,1) == doctest::Approx(     0.0998334166468));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(    -0.0998334166468));
                        CHECK(R(2,2) == doctest::Approx(      0.995004165278));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(                   0));
                        CHECK(dRdx(1,0) == doctest::Approx(                   0));
                        CHECK(dRdx(2,0) == doctest::Approx(                   0));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                   0));
                        CHECK(dRdx(1,1) == doctest::Approx(    -0.0998334166468));
                        CHECK(dRdx(2,1) == doctest::Approx(      0.995004165278));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   0));
                        CHECK(dRdx(1,2) == doctest::Approx(     -0.995004165278));
                        CHECK(dRdx(2,2) == doctest::Approx(    -0.0998334166468));
                    }
                }
            }
        }
    }

    GIVEN("x = pi*5/3")
    {
        Eigen::Matrix3d R;
        double x = M_PI*5/3;

        WHEN("Calling R = rotx(x)")
        {
            R = rotx(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                 0.5));
                        CHECK(R(2,1) == doctest::Approx(     -0.866025403784));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(      0.866025403784));
                        CHECK(R(2,2) == doctest::Approx(                 0.5));
                    }
                }
            }
        }

        WHEN("Calling R = rotx(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = rotx(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                 0.5));
                        CHECK(R(2,1) == doctest::Approx(     -0.866025403784));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(      0.866025403784));
                        CHECK(R(2,2) == doctest::Approx(                 0.5));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(                   0));
                        CHECK(dRdx(1,0) == doctest::Approx(                   0));
                        CHECK(dRdx(2,0) == doctest::Approx(                   0));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                   0));
                        CHECK(dRdx(1,1) == doctest::Approx(      0.866025403784));
                        CHECK(dRdx(2,1) == doctest::Approx(                 0.5));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   0));
                        CHECK(dRdx(1,2) == doctest::Approx(                -0.5));
                        CHECK(dRdx(2,2) == doctest::Approx(      0.866025403784));
                    }
                }
            }
        }
    }
}


SCENARIO("roty")
{
    GIVEN("x = 0")
    {
        Eigen::Matrix3d R;
        double x = 0;

        WHEN("Calling R = roty(x)")
        {
            R = roty(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                  -0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }
        }

        WHEN("Calling R = roty(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = roty(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                  -0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(                  -0));
                        CHECK(dRdx(1,0) == doctest::Approx(                   0));
                        CHECK(dRdx(2,0) == doctest::Approx(                  -1));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                   0));
                        CHECK(dRdx(1,1) == doctest::Approx(                   0));
                        CHECK(dRdx(2,1) == doctest::Approx(                   0));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   1));
                        CHECK(dRdx(1,2) == doctest::Approx(                   0));
                        CHECK(dRdx(2,2) == doctest::Approx(                  -0));
                    }
                }
            }
        }
    }

    GIVEN("x = 0.1")
    {

        Eigen::Matrix3d R;
        double x = 0.1;

        WHEN("Calling R = roty(x)")
        {
            R = roty(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(      0.995004165278));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(    -0.0998334166468));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(     0.0998334166468));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(      0.995004165278));
                    }
                }
            }
        }

        WHEN("Calling R = roty(x, dRdx)"){
            Eigen::Matrix3d dRdx;
            R = roty(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(      0.995004165278));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(    -0.0998334166468));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(     0.0998334166468));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(      0.995004165278));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty"){
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions"){
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);
                    AND_THEN("dRdx is correct"){

                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(    -0.0998334166468));
                        CHECK(dRdx(1,0) == doctest::Approx(                   0));
                        CHECK(dRdx(2,0) == doctest::Approx(     -0.995004165278));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                   0));
                        CHECK(dRdx(1,1) == doctest::Approx(                   0));
                        CHECK(dRdx(2,1) == doctest::Approx(                   0));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(      0.995004165278));
                        CHECK(dRdx(1,2) == doctest::Approx(                   0));
                        CHECK(dRdx(2,2) == doctest::Approx(    -0.0998334166468));

                    }
                }
            }        
        }
    }

    GIVEN("x = pi*5/3")
    {
        Eigen::Matrix3d R;
        double x = M_PI*5/3;

        WHEN("Calling R = roty(x)")
        {
            R = roty(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                 0.5));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(      0.866025403784));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(     -0.866025403784));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                 0.5));
                    }
                }
            }
        }

        WHEN("Calling R = roty(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = roty(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                 0.5));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(      0.866025403784));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(     -0.866025403784));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                 0.5));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(      0.866025403784));
                        CHECK(dRdx(1,0) == doctest::Approx(                   0));
                        CHECK(dRdx(2,0) == doctest::Approx(                -0.5));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                   0));
                        CHECK(dRdx(1,1) == doctest::Approx(                   0));
                        CHECK(dRdx(2,1) == doctest::Approx(                   0));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                 0.5));
                        CHECK(dRdx(1,2) == doctest::Approx(                   0));
                        CHECK(dRdx(2,2) == doctest::Approx(      0.866025403784));

                    }
                }
            }
        }
    }
}

SCENARIO("rotz")
{
    GIVEN("x = 0")
    {
        Eigen::Matrix3d R;
        double x = 0;

        WHEN("Calling R = rotz(x)")
        {
            R = rotz(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                   0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }
        }

        WHEN("Calling R = rotz(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = rotz(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                   1));
                        CHECK(R(1,0) == doctest::Approx(                   0));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(                  -0));
                        CHECK(R(1,1) == doctest::Approx(                   1));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(                  -0));
                        CHECK(dRdx(1,0) == doctest::Approx(                   1));
                        CHECK(dRdx(2,0) == doctest::Approx(                   0));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                  -1));
                        CHECK(dRdx(1,1) == doctest::Approx(                  -0));
                        CHECK(dRdx(2,1) == doctest::Approx(                   0));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   0));
                        CHECK(dRdx(1,2) == doctest::Approx(                   0));
                        CHECK(dRdx(2,2) == doctest::Approx(                   0));
                    }
                }
            }
        }
    }

    GIVEN("x = 0.1")
    {
        Eigen::Matrix3d R;
        double x = 0.1;

        WHEN("Calling R = rotz(x)")
        {
            R = rotz(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions"){

                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(      0.995004165278));
                        CHECK(R(1,0) == doctest::Approx(     0.0998334166468));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(    -0.0998334166468));
                        CHECK(R(1,1) == doctest::Approx(      0.995004165278));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }
        }

        WHEN("Calling R = rotz(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = rotz(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(      0.995004165278));
                        CHECK(R(1,0) == doctest::Approx(     0.0998334166468));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(    -0.0998334166468));
                        CHECK(R(1,1) == doctest::Approx(      0.995004165278));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(    -0.0998334166468));
                        CHECK(dRdx(1,0) == doctest::Approx(      0.995004165278));
                        CHECK(dRdx(2,0) == doctest::Approx(                   0));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(     -0.995004165278));
                        CHECK(dRdx(1,1) == doctest::Approx(    -0.0998334166468));
                        CHECK(dRdx(2,1) == doctest::Approx(                   0));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   0));
                        CHECK(dRdx(1,2) == doctest::Approx(                   0));
                        CHECK(dRdx(2,2) == doctest::Approx(                   0));
                    }
                }
            }
        }
    }

    GIVEN("x = pi*5/3")
    {
        Eigen::Matrix3d R;
        double x = M_PI*5/3;

        WHEN("Calling R = rotz(x)")
        {
            R = rotz(x);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                 0.5));
                        CHECK(R(1,0) == doctest::Approx(     -0.866025403784));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(      0.866025403784));
                        CHECK(R(1,1) == doctest::Approx(                 0.5));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }
        }

        WHEN("Calling R = rotz(x, dRdx)")
        {
            Eigen::Matrix3d dRdx;
            R = rotz(x, dRdx);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(                 0.5));
                        CHECK(R(1,0) == doctest::Approx(     -0.866025403784));
                        CHECK(R(2,0) == doctest::Approx(                   0));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(      0.866025403784));
                        CHECK(R(1,1) == doctest::Approx(                 0.5));
                        CHECK(R(2,1) == doctest::Approx(                   0));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(                   0));
                        CHECK(R(1,2) == doctest::Approx(                   0));
                        CHECK(R(2,2) == doctest::Approx(                   1));
                    }
                }
            }

            //--------------------------------------------------------------------------------
            // Checks for dRdx 
            //--------------------------------------------------------------------------------
            THEN("dRdx is not empty")
            {
                REQUIRE(dRdx.size()>0);
                
                AND_THEN("dRdx has the right dimensions")
                {
                    REQUIRE(dRdx.rows()==3);
                    REQUIRE(dRdx.cols()==3);

                    AND_THEN("dRdx is correct")
                    {
                        // dRdx(:,1)
                        CHECK(dRdx(0,0) == doctest::Approx(      0.866025403784));
                        CHECK(dRdx(1,0) == doctest::Approx(                 0.5));
                        CHECK(dRdx(2,0) == doctest::Approx(                   0));

                        // dRdx(:,2)
                        CHECK(dRdx(0,1) == doctest::Approx(                -0.5));
                        CHECK(dRdx(1,1) == doctest::Approx(      0.866025403784));
                        CHECK(dRdx(2,1) == doctest::Approx(                   0));

                        // dRdx(:,3)
                        CHECK(dRdx(0,2) == doctest::Approx(                   0));
                        CHECK(dRdx(1,2) == doctest::Approx(                   0));
                        CHECK(dRdx(2,2) == doctest::Approx(                   0));
                    }
                }
            }
        }
    }
}

SCENARIO("rpy2rot")
{
    GIVEN("Theta = [1; 2; 3]")
    {
        Eigen::Matrix3d R;
        Eigen::Vector3d Theta;
        // Theta - [3 x 1]: 
        Theta <<                    1,
                                    2,
                                    3;

        WHEN("Calling R = rpy2rot(Theta)")
        {
            R = rpy2rot(Theta);
            //--------------------------------------------------------------------------------
            // Checks for R 
            //--------------------------------------------------------------------------------
            THEN("R is not empty")
            {
                REQUIRE(R.size()>0);
                
                AND_THEN("R has the right dimensions")
                {
                    REQUIRE(R.rows()==3);
                    REQUIRE(R.cols()==3);

                    AND_THEN("R is correct")
                    {
                        // R(:,1)
                        CHECK(R(0,0) == doctest::Approx(      0.411982245666));
                        CHECK(R(1,0) == doctest::Approx(    -0.0587266449276));
                        CHECK(R(2,0) == doctest::Approx(     -0.909297426826));

                        // R(:,2)
                        CHECK(R(0,1) == doctest::Approx(     -0.833737651774));
                        CHECK(R(1,1) == doctest::Approx(     -0.426917621276));
                        CHECK(R(2,1) == doctest::Approx(     -0.350175488374));

                        // R(:,3)
                        CHECK(R(0,2) == doctest::Approx(     -0.367630462925));
                        CHECK(R(1,2) == doctest::Approx(      0.902381585483));
                        CHECK(R(2,2) == doctest::Approx(     -0.224845095366));
                    }
                }
            }
        }
    }
}

SCENARIO("rot2rpy")
{
    GIVEN("R is identity")
    {
        Eigen::Matrix3d R(3,3);
        Eigen::Vector3d Theta;

        // R - [3 x 3]: 
        R <<                    1,                 0,                 0,
                                0,                 1,                 0,
                                0,                 0,                 1;
        WHEN("Calling Theta = rot2rpy(R)")
        {
            Theta = rot2rpy(R);
            //--------------------------------------------------------------------------------
            // Checks for Theta 
            //--------------------------------------------------------------------------------
            THEN("Theta is not empty")
            {
                REQUIRE(Theta.size()>0);
                
                AND_THEN("Theta has the right dimensions")
                {
                    REQUIRE(Theta.rows()==3);
                    REQUIRE(Theta.cols()==1);

                    AND_THEN("Theta is correct")
                    {
                        // Theta(:,1)
                        CHECK(Theta(0,0) == doctest::Approx(                   0));
                        CHECK(Theta(1,0) == doctest::Approx(                  -0));
                        CHECK(Theta(2,0) == doctest::Approx(                   0));
                    }
                }
            }
        }
    }

    GIVEN("R is rotation about x axis by 0.5 rad")
    {
        Eigen::Matrix3d R(3,3);
        Eigen::Vector3d Theta;

        // R - [3 x 3]: 
        R <<                    1,                 0,                 0,
                                0,      0.8775825619,     -0.4794255386,
                                0,      0.4794255386,      0.8775825619;
        WHEN("Calling Theta = rot2rpy(R)")
        {
            Theta = rot2rpy(R);
            //--------------------------------------------------------------------------------
            // Checks for Theta 
            //--------------------------------------------------------------------------------
            THEN("Theta is not empty")
            {
                REQUIRE(Theta.size()>0);
                
                AND_THEN("Theta has the right dimensions")
                {
                    REQUIRE(Theta.rows()==3);
                    REQUIRE(Theta.cols()==1);

                    AND_THEN("Theta is correct")
                    {
                        // Theta(:,1)
                        CHECK(Theta(0,0) == doctest::Approx(                 0.5));
                        CHECK(Theta(1,0) == doctest::Approx(                  -0));
                        CHECK(Theta(2,0) == doctest::Approx(                   0));
                    }
                }
            }
        }
    }

    GIVEN("R is rotation about y axis by 0.5 rad")
    {
        Eigen::Matrix3d R(3,3);
        Eigen::Vector3d Theta;

        // R - [3 x 3]: 
        R <<         0.8775825619,                 0,      0.4794255386,
                                0,                 1,                 0,
                    -0.4794255386,                 0,      0.8775825619;
        WHEN("Calling Theta = rot2rpy(R)")
        {
            Theta = rot2rpy(R);
            //--------------------------------------------------------------------------------
            // Checks for Theta 
            //--------------------------------------------------------------------------------
            THEN("Theta is not empty")
            {
                REQUIRE(Theta.size()>0);
                
                AND_THEN("Theta has the right dimensions")
                {
                    REQUIRE(Theta.rows()==3);
                    REQUIRE(Theta.cols()==1);

                    AND_THEN("Theta is correct")
                    {
                        // Theta(:,1)
                        CHECK(Theta(0,0) == doctest::Approx(                   0));
                        CHECK(Theta(1,0) == doctest::Approx(                 0.5));
                        CHECK(Theta(2,0) == doctest::Approx(                   0));
                    }
                }
            }
        }
    }

    GIVEN("R is rotation about z axis by 0.5 rad")
    {
        Eigen::Matrix3d R(3,3);
        Eigen::Vector3d Theta;

        // R - [3 x 3]: 
        R <<         0.8775825619,     -0.4794255386,                 0,
                     0.4794255386,      0.8775825619,                 0,
                                0,                 0,                 1;
        WHEN("Calling Theta = rot2rpy(R)")
        {
            Theta = rot2rpy(R);
            //--------------------------------------------------------------------------------
            // Checks for Theta 
            //--------------------------------------------------------------------------------
            THEN("Theta is not empty")
            {
                REQUIRE(Theta.size()>0);
                
                AND_THEN("Theta has the right dimensions")
                {
                    REQUIRE(Theta.rows()==3);
                    REQUIRE(Theta.cols()==1);

                    AND_THEN("Theta is correct")
                    {
                        // Theta(:,1)
                        CHECK(Theta(0,0) == doctest::Approx(                   0));
                        CHECK(Theta(1,0) == doctest::Approx(                  -0));
                        CHECK(Theta(2,0) == doctest::Approx(                 0.5));
                    }
                }
            }
        }
    }

    GIVEN("R is a general rotation")
    {
        Eigen::Matrix3d R(3,3);
        Eigen::Vector3d Theta;

        // R - [3 x 3]: 
        R <<         0.4119822457,     -0.8337376518,     -0.3676304629,
                   -0.05872664493,     -0.4269176213,      0.9023815855,
                    -0.9092974268,     -0.3501754884,     -0.2248450954;
        WHEN("Calling Theta = rot2rpy(R)")
        {
            Theta = rot2rpy(R);
            //--------------------------------------------------------------------------------
            // Checks for Theta 
            //--------------------------------------------------------------------------------
            THEN("Theta is not empty")
            {
                REQUIRE(Theta.size()>0);
                Eigen::Matrix3d R_act = rpy2rot(Theta);

                AND_THEN("rot2rpy is an inverse mapping")
                {
                    // R_act(:,1)
                    CHECK(R_act(0,0) == doctest::Approx(R(0,0)));
                    CHECK(R_act(1,0) == doctest::Approx(R(1,0)));
                    CHECK(R_act(2,0) == doctest::Approx(R(2,0)));

                    // R_act(:,2)
                    CHECK(R_act(0,1) == doctest::Approx(R(0,1)));
                    CHECK(R_act(1,1) == doctest::Approx(R(1,1)));
                    CHECK(R_act(2,1) == doctest::Approx(R(2,1)));

                    // R_act(:,3)
                    CHECK(R_act(0,2) == doctest::Approx(R(0,2)));
                    CHECK(R_act(1,2) == doctest::Approx(R(1,2)));
                    CHECK(R_act(2,2) == doctest::Approx(R(2,2)));
                }
            }
        }
    }
}

SCENARIO("eulerKinematicTransformation")
{
    // Helper to build T(Theta) directly (for cross-checking the eulerKinematicTransformation bottom-right block)
    auto T_from = [](const Eigen::Vector3d& Theta) {
        return TK(Theta);
    };

    GIVEN("eta = [0, 0, 0, 0, 0, 0] (identity)")
    {
        Eigen::Matrix<double,6,1> eta;
        eta << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;  // [position; attitude]
        Eigen::Matrix<double,6,6> J = eulerKinematicTransformation(eta);

        THEN("J has correct size and block structure")
        {
            REQUIRE(J.rows() == 6);
            REQUIRE(J.cols() == 6);

            // Top-left should be identity (Rnb = I at zero angles)
            CHECK(J.block<3,3>(0,0)(0,0) == doctest::Approx(1.0));
            CHECK(J.block<3,3>(0,0)(1,1) == doctest::Approx(1.0));
            CHECK(J.block<3,3>(0,0)(2,2) == doctest::Approx(1.0));
            CHECK(J.block<3,3>(0,0)(0,1) == doctest::Approx(0.0));
            CHECK(J.block<3,3>(0,0)(0,2) == doctest::Approx(0.0));
            CHECK(J.block<3,3>(0,0)(1,0) == doctest::Approx(0.0));
            CHECK(J.block<3,3>(0,0)(1,2) == doctest::Approx(0.0));
            CHECK(J.block<3,3>(0,0)(2,0) == doctest::Approx(0.0));
            CHECK(J.block<3,3>(0,0)(2,1) == doctest::Approx(0.0));

            // Off-diagonal blocks should be zero
            CHECK(J.block<3,3>(0,3).norm() == doctest::Approx(0.0));
            CHECK(J.block<3,3>(3,0).norm() == doctest::Approx(0.0));

            // Bottom-right should be identity at zero angles
            const auto T = J.block<3,3>(3,3);
            CHECK(T(0,0) == doctest::Approx(1.0));
            CHECK(T(1,1) == doctest::Approx(1.0));
            CHECK(T(2,2) == doctest::Approx(1.0));
            CHECK(T(0,1) == doctest::Approx(0.0));
            CHECK(T(0,2) == doctest::Approx(0.0));
            CHECK(T(1,0) == doctest::Approx(0.0));
            CHECK(T(1,2) == doctest::Approx(0.0));
            CHECK(T(2,0) == doctest::Approx(0.0));
            CHECK(T(2,1) == doctest::Approx(0.0));
        }
    }

    GIVEN("General angles eta = [0, 0, 0, 0.3, -0.4, 0.5]")
    {
        Eigen::Matrix<double,6,1> eta;
        eta << 0.0, 0.0, 0.0, 0.3, -0.4, 0.5;  // [position; attitude]
        Eigen::Matrix<double,6,6> J = eulerKinematicTransformation(eta);

        THEN("Top-left equals rpy2rot(Theta) and is orthonormal")
        {
            Eigen::Vector3d Theta = eta.segment<3>(3);  // Extract attitude angles
            Eigen::Matrix3d R_expected = rpy2rot(Theta);
            Eigen::Matrix3d R_block    = J.block<3,3>(0,0);

            // Entry-wise comparison
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    CHECK(R_block(r,c) == doctest::Approx(R_expected(r,c)));

            // Orthonormality check: R * R^T = I
            Eigen::Matrix3d I = R_block * R_block.transpose();
            CHECK(I(0,0) == doctest::Approx(1.0));
            CHECK(I(1,1) == doctest::Approx(1.0));
            CHECK(I(2,2) == doctest::Approx(1.0));
            CHECK(I(0,1) == doctest::Approx(0.0));
            CHECK(I(0,2) == doctest::Approx(0.0));
            CHECK(I(1,0) == doctest::Approx(0.0));
            CHECK(I(1,2) == doctest::Approx(0.0));
            CHECK(I(2,0) == doctest::Approx(0.0));
            CHECK(I(2,1) == doctest::Approx(0.0));
        }

        THEN("Bottom-right equals T(Theta) from the closed-form expression")
        {
            Eigen::Vector3d Theta = eta.segment<3>(3);  // Extract attitude angles
            Eigen::Matrix3d T_expected = T_from(Theta);
            Eigen::Matrix3d T_block    = J.block<3,3>(3,3);

            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    CHECK(T_block(r,c) == doctest::Approx(T_expected(r,c)));
        }

        THEN("Off-diagonal 3x3 blocks are zero")
        {
            CHECK(J.block<3,3>(0,3).norm() == doctest::Approx(0.0));
            CHECK(J.block<3,3>(3,0).norm() == doctest::Approx(0.0));
        }

        THEN("Consistent mapping: eta_dot = J * nu matches blockwise computation")
        {
            // Pick a representative body velocity vector nu = [u v w p q r]
            Eigen::Matrix<double,6,1> nu;
            nu << 0.7, -1.2, 0.5,  // linear (body)
                  0.3,  0.1, -0.4; // angular (body)

            Eigen::Matrix<double,6,1> eta_dot = J * nu;

            // Extract attitude angles for computation
            Eigen::Vector3d Theta = eta.segment<3>(3);

            // Top: x_dot = Rnb * v_body
            Eigen::Vector3d v_body = nu.head<3>();
            Eigen::Vector3d xdot_expected = rpy2rot(Theta) * v_body;

            // Bottom: euler_dot = T(Theta) * omega_body
            Eigen::Vector3d w_body = nu.tail<3>();
            Eigen::Vector3d eulerdot_expected = T_from(Theta) * w_body;

            for (int i = 0; i < 3; ++i) {
                CHECK(eta_dot(i)     == doctest::Approx(xdot_expected(i)));
                CHECK(eta_dot(3 + i) == doctest::Approx(eulerdot_expected(i)));
            }
        }
    }

    GIVEN("Near the pitch singularity: theta -> +pi/2")
    {
        const double eps = 1e-6;
        Eigen::Matrix<double,6,1> eta;
        eta << 0.0, 0.0, 0.0, 0.2, M_PI/2.0 - eps, -0.7;  // [position; attitude]
        Eigen::Matrix<double,6,6> J = eulerKinematicTransformation(eta);

        THEN("T(Theta) entries reflect large tan(theta) terms but are finite for theta = pi/2 - eps")
        {
            Eigen::Matrix3d T_block = J.block<3,3>(3,3);

            // We don't assert exact numbers (they can be large),
            // but we check they are finite and the expected elements are nontrivial
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    CHECK(std::isfinite(T_block(r,c)));

            // The (0,1) and (0,2) elements contain tan(theta), so they should be "large"
            CHECK(std::abs(T_block(0,1)) > 1e3);
            CHECK(std::abs(T_block(0,2)) > 1e3);
        }
    }
}

