#include <bits/stdc++.h>
using namespace std;

int main(){
    int n, m;
    int num;
    cin >> n >> m;
    num = n*(n-1) / 2;
    num += m*(m-1) / 2;
    cout << num << endl;
}