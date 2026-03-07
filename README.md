# 🤖 Production-Grade GenAI Assistant with RAG

A professional implementation of a Retrieval-Augmented Generation (RAG) system for question answering, built with Python, Flask, and OpenAI APIs.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [RAG Workflow](#rag-workflow)
- [Deployment](#deployment)
- [Project Structure](#project-structure)

## 🎯 Overview

This project implements a production-grade chat assistant that uses **Retrieval-Augmented Generation (RAG)** to provide accurate, source-grounded answers to user questions. Unlike simple chatbots, this system:

1. **Retrieves** relevant context from a knowledge base using semantic search
2. **Augments** the LLM prompt with retrieved context
3. **Generates** responses grounded in actual documentation

## 🏗 Architecture