#include <bits/stdc++.h>
using namespace std;

int main(){
    int h, w;
    cin >> h >> w;
    vector<vector <char>> a(h, vector<char>(w));
    vector<int> b(h), c(w);

    for (int i=0; i<h; i++){
        for (int j=0; j<w; j++){
            cin >> a.at(i).at(j);
        }
    }

    int flg;
    for (int i=0; i<h; i++){
        int flg = 1;
        for (int j=0; j<w; j++){
            if (a.at(i).at(j) == '.'){
                flg *= 1;
            }else{
                flg *= 0;
            }
        }
        b.at(i) = flg;
    }

    for (int j=0; j<w; j++){
        int flg = 1;
        for (int i=0; i<h; i++){
            if (a.at(i).at(j) == '.'){
                flg *= 1;
            }else{
                flg *= 0;
            }
        }
        c.at(j) = flg;
    }

    for(int i=0; i<h; i++){
        if (b.at(i) == 1){
            continue;
        }
        for (int j=0; j<w; j++){
            if(c.at(j) == 0){
                cout << a.at(i).at(j);
            }
        }
        cout << endl;
    }
}