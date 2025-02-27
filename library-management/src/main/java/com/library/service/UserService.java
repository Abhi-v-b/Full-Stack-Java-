package com.library.service;

import com.library.entity.User;
import java.util.List;
import java.util.Optional;

public interface UserService {
    User saveUser(User user);
    Optional<User> getUserById(Long id);
    Optional<User> getUserByEmail(String email);
    List<User> getAllUsers();
    List<User> searchUsers(String query);
    void deleteUser(Long id);
    User updateUser(Long id, User user);
    boolean existsByEmail(String email);
    List<User> getActiveUsers();
    void deactivateUser(Long id);
    void activateUser(Long id);
} 