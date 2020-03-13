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

int insertionSort(vector<int> &a, int &n, int g){
    int cnt = 0;
    for(int i=g; i<n; i++){
        int v = a.at(i);
        int j = i - g;
        while(j>=0 && a.at(j) > v){
            int b = a.at(j);
            int c = a.at(j+g);
            a.at(j+g) = b;
            a.at(j) = c;
            j = j - g;
            cnt++;
        }
        a.at(j+g) = v;
    }
    return cnt;
}

int myArr(int n){
    if(n==0){
        return 1;
    }
    return myArr(n-1) * 3 + 1;
}

int shellSort(vector<int> &a, int &n){
    int cnt = 0;
    vector<int> g;
    for (int i=1; ; ){
        if(i > n) break;
        g.push_back(i);
        i = 3 * i + 1;
    }
    int m = g.size();

    for (int i=m-1; i>=0; i--){
        cnt += insertionSort(a, n, g.at(i));
    }
    cout << m << endl;
    for (int i=m-1; i>=0; i--){
        cout << g.at(i);
        if (i>0){
            cout << " ";
        }else{
            cout << endl;
        }
    }
    return cnt;

}

int main(){
    int n, cnt;
    cin >> n;
    vector<int> a;
    for(int i=0; i<n; i++){
        int b;
        cin >> b;
        a.push_back(b);
    }
    cnt = shellSort(a, n);
    cout << cnt << endl;
    for(int i=0; i<n; i++){
        cout << a.at(i) << endl;
    }
}
