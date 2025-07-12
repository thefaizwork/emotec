import axios from 'axios';

const baseURL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export const api = axios.create({ baseURL });

export const startSession = () => api.post('/combine/start');
export const stopSession = () => api.post('/combine/stop');
export const healthCheck = () => api.get('/health');

export const listSessions = () => api.get('/sessions');
export const getSession = (id: string) => api.get(`/sessions/${id}`);
export const deleteSession = (id: string) => api.delete(`/sessions/${id}`);
