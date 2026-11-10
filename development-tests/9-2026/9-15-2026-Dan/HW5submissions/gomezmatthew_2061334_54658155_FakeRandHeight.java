/*

  Author: Phil Chan
  Email: pkc@cs.fit.edu
  Description:  

  For easier debugging and testing, this class provides a known
  sequence of fake random height for HW5 in CSE2010.  The sequence
  simulates each successive higher level is half as likely to be used
  in SkipListMap.put().

  The sequence has 31 numbers: 0 occurs 16 times, 1 occurs 8 times, 2
  occurs 4 times, 3 occurs 2 times, and 4 occurs 1 time.  Each number
  is returned by get() in the following order and the numbers are
  recycled if get() is called more than 31 times.

                                1 1 1 1 1 1
  index:    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5

  sequence: 0 1 0 2 0 1 0 3 0 1 0 2 0 1 0 4

            1 1 1 1 2 2 2 2 2 2 2 2 2 2 3
  index:    6 7 8 9 0 1 2 3 4 5 6 7 8 9 0

  sequence: 0 1 0 2 0 1 0 3 0 1 0 2 0 1 0  


  Usage:

  FakeRandHeight randHeight = new FakeRandHeight();

  randHeight.get()  

 */

public class FakeRandHeight
{
    static int[] height = {0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0, 4, // sequence of height
                           0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0, 2, 0, 1, 0}; 
    int count;  // number of times get() has been called 

    public FakeRandHeight()
    {
	count = 0;
    }

    // return the next number in the sequence, recycle the sequence if needed
    public int get()
    {
	return height[count++ % 31];
    }

    // testing get()
    /*
    public static void main(String[] args)
    {
	FakeRandHeight randHeight = new FakeRandHeight();

	for (int i=0; i < 40; i++)
	    {
		System.out.println(i + " " + randHeight.get());
	    }
    }
    */
}
