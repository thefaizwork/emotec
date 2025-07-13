import axios from 'axios';

const baseURL = 'https://emotec.onrender.com';

export const api = axios.create({ baseURL });

export const startSession = () => api.post('/combine/start');
export const stopSession = () => api.post('/combine/stop');
export const healthCheck = () => api.get('/health');

export const listSessions = () => api.get('/sessions');
export const getSession = (id: string) => api.get(`/sessions/${id}`);
export const deleteSession = (id: string) => api.delete(`/sessions/${id}`);
