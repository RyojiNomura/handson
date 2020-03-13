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

void selectionSort(vector<int> &a, int &n){
    int cnt = 0;
    for(int i=0; i<n; i++){
        int min = a.at(i);
        int loc = i;
        for(int j=i; j<n; j++){
            if(min > a.at(j)){
                min = a.at(j);
                loc = j;
            }
        }
        if(loc != i){
            cnt++;
            a.at(loc) = a.at(i);
            a.at(i) = min;
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
    selectionSort(a, n);

}
