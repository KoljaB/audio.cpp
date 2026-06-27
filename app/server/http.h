#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace minitts::server {

struct HttpRequest {
    std::string method;
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
};

struct HttpResponse {
    int status = 200;
    std::string content_type = "application/json";
    std::string body;
    std::unordered_map<std::string, std::string> headers;
};

class HttpResponder {
public:
    explicit HttpResponder(std::uintptr_t socket);

    HttpResponder(const HttpResponder &) = delete;
    HttpResponder & operator=(const HttpResponder &) = delete;

    bool headers_sent() const noexcept;
    void send_response(
        int status,
        std::string content_type,
        std::string body,
        std::unordered_map<std::string, std::string> headers = {});
    void start_chunked(
        int status,
        std::string content_type,
        std::unordered_map<std::string, std::string> headers = {});
    void send_chunk(std::string_view data);
    void finish_chunked();

private:
    std::uintptr_t socket_ = 0;
    bool headers_sent_ = false;
    bool chunked_ = false;
};

class IHttpHandler {
public:
    virtual ~IHttpHandler() = default;
    virtual bool handle_stream(const HttpRequest & request, HttpResponder & responder);
    virtual HttpResponse handle(const HttpRequest & request) = 0;
};

HttpResponse json_response(std::string body, int status = 200);
HttpResponse error_response(int status, const std::string & message, const std::string & type);
void serve_http(const std::string & host, int port, IHttpHandler & handler);

}  // namespace minitts::server
