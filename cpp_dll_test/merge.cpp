#include "store.h"
#include "book.h"
#include "rele.h"
#include <iostream>

void SendMessage()
{
	Book Book1;
	Store Store1;
	Book1.SetPrice(10);
	Store1.SetBookNum(100);
	std::cout << Book1.GetPrice() << std::endl;
	std::cout << Store1.GetBookNum() << std::endl;
}