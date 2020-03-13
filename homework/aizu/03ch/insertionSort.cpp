#include<iostream>
#include<vector>
using namespace std;

void trace(vector<int> &a, int &n){
    for(int i=0; i<n; i++){
        cout << a.at(i);
        if(i<n-1){
            cout << " ";
        }
    }
    cout << endl;
}

void insertionSort(vector<int> &a, int &n){
    for(int i=1; i<n; i++){
        int v = a.at(i);
        int j = i - 1;
        while(a.at(j) > v){
            int b = a.at(j);
            int c = a.at(j+1);
            a.at(j+1) = b;
            a.at(j) = c;
            j--;
            if (j < 0){
                break;
            }
        }
        a.at(j+1) = v;
        trace(a, n);
    }
}

int main(){
    int n;
    cin >> n;
    vector<int> a;
    for(int i=0; i<n; i++){
        int b;
        cin >> b;
        a.push_back(b);
    }
    trace(a, n);
    insertionSort(a, n);

}
