#ifndef BOOK_H_
#define BOOK_H_


class Book
{
public:
	int GetPrice();
	void SetPrice(int NewPrice);
private:
	int Price = 0;
};

#endif // !BOOK_H_

