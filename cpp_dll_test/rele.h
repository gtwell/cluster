#ifndef RELE_H_
#define RELE_H_

#ifdef RELEDLL_EXPORTS
#   define RELEDLL_API __declspec(dllexport)
#else
#   define RELEDLL_API __declspec(dllimport)
#endif

RELEDLL_API void SendMessage();

#endif // RELE_H_
