#ifndef STORE_H_
#define STORE_H_

class Store
{
public:
	int GetBookNum();
	void SetBookNum(int NewBookNum);
private:
	int BookNum = 0;
};
#endif // !STORE_H_
