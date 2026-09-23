"""The model gateway: one interface in front of every language model.

Adapters translate the provider-neutral types in :mod:`synthia.gateway.types`
to a provider's wire format. Everything above the gateway speaks only those
types and the :class:`~synthia.gateway.protocol.ChatModel` protocol.
"""
