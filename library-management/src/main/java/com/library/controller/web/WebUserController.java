package com.library.controller.web;

import com.library.entity.User;
import com.library.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.validation.BindingResult;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

import jakarta.validation.Valid;

@Controller
@RequestMapping("/users")
public class WebUserController {

    private final UserService userService;

    @Autowired
    public WebUserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping
    public String listUsers(Model model) {
        model.addAttribute("users", userService.getAllUsers());
        return "users/list";
    }

    @GetMapping("/create")
    public String createUserForm(Model model) {
        model.addAttribute("user", new User());
        return "users/form";
    }

    @PostMapping
    public String createUser(@Valid @ModelAttribute User user,
                           BindingResult result,
                           RedirectAttributes redirectAttributes) {
        if (result.hasErrors()) {
            return "users/form";
        }

        if (userService.existsByEmail(user.getEmail())) {
            result.rejectValue("email", "error.user", "Email already exists");
            return "users/form";
        }

        userService.saveUser(user);
        redirectAttributes.addFlashAttribute("successMessage", "User created successfully");
        return "redirect:/users";
    }

    @GetMapping("/{id}/edit")
    public String editUserForm(@PathVariable Long id, Model model) {
        return userService.getUserById(id)
                .map(user -> {
                    model.addAttribute("user", user);
                    return "users/form";
                })
                .orElse("redirect:/users");
    }

    @PostMapping("/{id}")
    public String updateUser(@PathVariable Long id,
                           @Valid @ModelAttribute User user,
                           BindingResult result,
                           RedirectAttributes redirectAttributes) {
        if (result.hasErrors()) {
            return "users/form";
        }

        userService.updateUser(id, user);
        redirectAttributes.addFlashAttribute("successMessage", "User updated successfully");
        return "redirect:/users";
    }

    @GetMapping("/{id}/delete")
    public String deleteUser(@PathVariable Long id, RedirectAttributes redirectAttributes) {
        userService.deleteUser(id);
        redirectAttributes.addFlashAttribute("successMessage", "User deleted successfully");
        return "redirect:/users";
    }

    @GetMapping("/{id}/activate")
    public String activateUser(@PathVariable Long id, RedirectAttributes redirectAttributes) {
        userService.activateUser(id);
        redirectAttributes.addFlashAttribute("successMessage", "User activated successfully");
        return "redirect:/users";
    }

    @GetMapping("/{id}/deactivate")
    public String deactivateUser(@PathVariable Long id, RedirectAttributes redirectAttributes) {
        userService.deactivateUser(id);
        redirectAttributes.addFlashAttribute("successMessage", "User deactivated successfully");
        return "redirect:/users";
    }

    @GetMapping("/{id}")
    public String viewUser(@PathVariable Long id, Model model) {
        return userService.getUserById(id)
                .map(user -> {
                    model.addAttribute("user", user);
                    return "users/view";
                })
                .orElse("redirect:/users");
    }
} 