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

void bubbleSort(vector<int> &a, int &n){
    int cnt = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n-1; j++){
            int v = a.at(j);
            int w = a.at(j+1);
            if ( v > w){
                a.at(j) = w; 
                a.at(j+1) = v;
                cnt++; 
            }

        }
    }

    trace(a, n);
    cout << cnt << endl;

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
    bubbleSort(a, n);

}
