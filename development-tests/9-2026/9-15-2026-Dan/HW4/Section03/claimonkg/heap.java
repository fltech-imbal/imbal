import java.util.ArrayList;

public class heap {
    public ArrayList<Entry> heap = new ArrayList<>(); 

    public int parent(int j) {
        return (j - 1)/2;
    } 

    public int left(int j) {
        return 2*j + 1;
    } 

    public int right(int j) {
        return 2*j + 2;
    } 

    public boolean hasLeft(int j) {
        return left(j) < heap.size();
    } 

    public boolean hasRight(int j) {
        return right(j) < heap.size();
    }

    public boolean isEmpty() {
        if (heap.size() > 0) {
            return false;
        } else {
            return true;
        }
        
    } 

    public boolean isFull() {
        if (heap.size() > 100) {
            return true;
        } else {
            return false;
        }
        
    }  

    public void swap(int i, int j) {
        Entry temp = heap.get(i); 
        heap.set(i, heap.get(j)); 
        heap.set(j, temp);
    }  

    public boolean compare(Entry w, Entry y) {
        if(w.getKey() < y.getKey()) {
            return true;
        } else {
            return false;
        }

    }

    public void upheap(int j) {
        while (j > 0) {
            int p = parent(j);  
            if (compare(heap.get(j), heap.get(p)) == true) {
                swap(j, p); 
                j=p;
            } 

        }
    } 

    public void downheap(int j) {
        while(hasLeft(j)) {
            int leftIndex = left(j); 
            int smallChildIndex = leftIndex; 
            if(hasRight(j)) {
                int rightIndex = right(j); 
                if(compare(heap.get(rightIndex), heap.get(leftIndex)) == true) {
                    smallChildIndex = rightIndex;    //if rightIndex is smaller than leftIndex
                } 

            } 

            if(compare(heap.get(j), heap.get(smallChildIndex)) == true) {
                break;
            } 
            swap(j, smallChildIndex); 
            j = smallChildIndex;
        }
    } 

    public int size() {return heap.size(); }

    public void insert(Entry entry) {
        
        
        if (isFull() == true) {
            return;
        } else {
            heap.add(entry); 
            upheap(heap.indexOf(entry));

            
        }
    } 


    //public getMin() 
    public Entry getMin() {
        if (heap.isEmpty()) {
            return null;
        } else {
            return heap.get(0);
        }

    }

    //public removeMin() 
    public void removeMin() {
        if(heap.isEmpty()) {
            return;
        } else {
            swap(0, heap.size()-1); 
            heap.remove(heap.size()-1); 
            downheap(0);
        }

    } 

     

    

}