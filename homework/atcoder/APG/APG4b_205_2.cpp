#include <bits/stdc++.h>
using namespace std;


void root(vector<vector<char>> &map, vector<vector<bool>> &checked, int y, int x, int &h, int &w){
    if(x<0 || w<=x || y<0 || h<=y) return;
    if(map.at(y).at(x) == '#') return;
    if(checked.at(y).at(x) == true) return;

    checked.at(y).at(x) = true;
    root(map, checked, y, x-1, h, w);
    root(map, checked, y-1, x, h, w);
    root(map, checked, y, x+1, h, w);
    root(map, checked, y+1, x, h, w);
}


int main(){
    int h, w;
    cin >> h >> w;
    vector<vector<char>> map(h, vector<char>(w));
    vector<vector<bool>> checked(h, vector<bool>(w));
    int init_y, init_x;
    int goal_y, goal_x;

    for (int i=0; i<h; i++){
        for (int j=0; j<w; j++){
            cin >> map.at(i).at(j);
            if (map.at(i).at(j) == 's'){
                init_y = i;
                init_x = j;
            }
            if (map.at(i).at(j) == 'g'){
                goal_y = i;
                goal_x = j;
            }
        }
    }

    root(map, checked, init_y, init_x, h, w);

    if(checked.at(goal_y).at(goal_x) == true){
        cout << "Yes" << endl;
    }else{
        cout << "No" << endl;
    }

}