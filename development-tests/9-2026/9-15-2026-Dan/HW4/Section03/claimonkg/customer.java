

public class customer {
    String name; 
    int[] ratings; 

    public customer(String name, int[] ratings) {
        this.name = name; 
        this.ratings = ratings;
    }  
    
    public int[] getRatings() {
        return ratings;
    } 

    public String getName() {
        return name;
    }

    public double getDistance(customer j) {
        //get the distance between targetCustomer and closestCustomer, with "s" being the number of
        //songs targetCustomer rated. S is the set of songs. If S is 0, break the program. 
        //A sumnation is a loop from s to S. What they want is 1/|S| + 1/|S| * a for loop from s to S
        //s to S: |targetCustomer rating(s) - closestCustomer rating(s)| 

         
        int bothSongs = 0; 
        int total = 0;

        

        for (int x = 0; x < this.ratings.length; x++ ) {
            if (this.ratings[x] != 0 && j.ratings[x] !=0) {
                bothSongs++; 

                int difference = Math.abs(this.ratings[x] - j.ratings[x]); 

                total += difference;

            }
        }

        if (bothSongs == 0) { return -1; } 

        return (double) (1+total)/bothSongs;

    }

    public String recommendSongs( customer j) {
        //loop around the arrays, count any songs that customer j has rated, but customer i had not. 

        int numSong = 0; 
        int rating = 0; 
        String songsRecommended = "";
        for (int x = 0; x < this.ratings.length; x++) {
            //if-statement for i==0 and j >= 4 
            if(this.ratings[x] == 0 && j.ratings[x] >=4) {
                //grab the number from customer j and put it as a atring for songsRecommended 
                rating = j.ratings[x]; 
                songsRecommended += "song"+numSong + " " + rating + " ";
            }
            //numSong increments 
            numSong++;
        } 

        return songsRecommended;

    } 

    
    
}
