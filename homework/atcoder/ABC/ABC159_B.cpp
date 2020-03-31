#include <bits/stdc++.h>
using namespace std;

bool isInv(string s){
    string inv;
    inv = s;
    reverse(inv.begin(), inv.end());
    bool t = (s == inv);
    return t;
}

int main(){
    string s;
    cin >> s;
    int n;
    n = s.size();
    string left_s = s.substr(0, (n-1)/2);
    string right_s = s.substr((n+1)/2, n);

    bool all, left, right;
    all = isInv(s);
    left = isInv(left_s);
    right = isInv(right_s);
    if(all * left * right) cout << "Yes" << endl;
    else cout << "No" << endl;

}