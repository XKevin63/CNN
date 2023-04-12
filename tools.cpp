#include"tools.h"


int PrintProgress(int now, int all, int before) {
	float fnow = static_cast<float>(now+1);
	float fall = static_cast<float>(all);
	float t = fnow / fall;
	int per = static_cast<int>(t * 100);
	if (per == before)
		return per;
	system("cls");
	cout << "¿ªÊ¼£º" << endl;
	for (int i = 0; i <= per; i++) {
		cout << "=";
	}
	for (int i = per + 1; i < 100; i++) {
		cout << " ";
	}
	cout << "|" << per << "%" << endl;
	return per;
}
