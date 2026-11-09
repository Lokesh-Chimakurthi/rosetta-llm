"""The parse and render functions for each API format, and its upstream path."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

from rosetta.codecs import anthropic as ac_codec
from rosetta.codecs import openai_chat as oc_codec
from rosetta.codecs import openai_responses as or_codec
from rosetta.config import Format
from rosetta.ir.request import CanonicalRequest
from rosetta.ir.response import CanonicalResponse
from rosetta.stream_codecs import anthropic as ac_stream
from rosetta.stream_codecs import openai_chat as oc_stream
from rosetta.stream_codecs import openai_responses as or_stream

type ParseRequest = Callable[[dict[str, Any]], CanonicalRequest]
type RenderRequest = Callable[[CanonicalRequest], dict[str, Any]]
type ParseResponse = Callable[[dict[str, Any]], CanonicalResponse]
type RenderResponse = Callable[[CanonicalResponse], dict[str, Any]]
type StreamParse = Callable[[AsyncIterator[bytes]], AsyncIterator[Any]]
type StreamRender = Callable[[AsyncIterator[Any]], AsyncIterator[bytes]]

PARSE_REQUEST: Mapping[Format, ParseRequest] = {
    "anthropic": ac_codec.parse_request,
    "openai_chat": oc_codec.parse_request,
    "openai_responses": or_codec.parse_request,
}
RENDER_REQUEST: Mapping[Format, RenderRequest] = {
    "anthropic": ac_codec.render_request,
    "openai_chat": oc_codec.render_request,
    "openai_responses": or_codec.render_request,
}
PARSE_RESPONSE: Mapping[Format, ParseResponse] = {
    "anthropic": ac_codec.parse_response,
    "openai_chat": oc_codec.parse_response,
    "openai_responses": or_codec.parse_response,
}
RENDER_RESPONSE: Mapping[Format, RenderResponse] = {
    "anthropic": ac_codec.render_response,
    "openai_chat": oc_codec.render_response,
    "openai_responses": or_codec.render_response,
}
STREAM_PARSE: Mapping[Format, StreamParse] = {
    "anthropic": ac_stream.parse,
    "openai_chat": oc_stream.parse,
    "openai_responses": or_stream.parse,
}
STREAM_RENDER: Mapping[Format, StreamRender] = {
    "anthropic": ac_stream.render,
    "openai_chat": oc_stream.render,
    "openai_responses": or_stream.render,
}
UPSTREAM_PATH: Mapping[Format, str] = {
    "openai_chat": "/chat/completions",
    "openai_responses": "/responses",
    "anthropic": "/messages",
}
