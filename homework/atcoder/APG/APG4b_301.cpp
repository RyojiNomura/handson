#include <bits/stdc++.h>
using namespace std;

int64_t luca(int n){
    if (n==0){
        return 2;
    }else if(n==1){
        return 1;
    }

    return luca(n-2) + luca(n-1);
}

int main(){
    int n;
    cin >> n;
    int64_t out = luca(n);

    cout << luca(n) << endl;

}